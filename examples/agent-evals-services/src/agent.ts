// A billing support agent for a metered API: it reads the customer's plan from
// Postgres and their usage from ClickHouse, and can credit their account.
//
//   node src/agent.ts case.json    # prints the reply to the case's ticket
import { readFileSync } from "node:fs";

import Anthropic from "@anthropic-ai/sdk";
import { createClient } from "@clickhouse/client";
import { Laminar, observe } from "@lmnr-ai/lmnr";
import pg from "pg";

Laminar.initialize({ instrumentModules: { anthropic: Anthropic } });

const MODEL = process.env.AGENT_MODEL ?? "claude-sonnet-5-5";
const db = new pg.Pool({ connectionString: process.env.DATABASE_URL });
const clickhouse = createClient({
  url: process.env.CLICKHOUSE_URL,
  database: process.env.CLICKHOUSE_DB,
});
const anthropic = new Anthropic();

const SYSTEM_PROMPT = `You are a billing support agent for a metered API.
Customers pay for every event recorded in the last 30 days above their plan's
included events, at their plan's overage rate. An event id recorded more than
once is an ingestion fault on our side: those duplicates must not be billed, so
credit the overage they caused (whole cents, rounded down) and nothing else.
Legitimate usage is never credited. Use the tools to check the account and its
usage before deciding. When you are done, write a short reply to the customer.`;

const TOOLS: Anthropic.Tool[] = [
  {
    name: "get_account",
    description: "Look up an account by the customer's email: plan, included events, overage rate.",
    input_schema: {
      type: "object",
      properties: { email: { type: "string" } },
      required: ["email"],
    },
  },
  {
    name: "get_usage",
    description:
      "An account's usage over the last 30 days, per day: events recorded and distinct event ids.",
    input_schema: {
      type: "object",
      properties: { account_id: { type: "string" } },
      required: ["account_id"],
    },
  },
  {
    name: "issue_credit",
    description: "Credit an account, in cents, with the reason shown on the customer's invoice.",
    input_schema: {
      type: "object",
      properties: {
        account_id: { type: "string" },
        amount_cents: { type: "integer" },
        reason: { type: "string" },
      },
      required: ["account_id", "amount_cents", "reason"],
    },
  },
];

const getAccount = async ({ email }: { email: string }) => {
  const { rows } = await db.query(
    "SELECT id, name, plan, included_events, overage_cents_per_1000 FROM accounts WHERE email = $1",
    [email],
  );
  return rows[0] ?? { error: `No account for ${email}` };
};

const getUsage = async ({ account_id }: { account_id: string }) => {
  const result = await clickhouse.query({
    query: `SELECT toDate(ts) AS day, count() AS events, uniqExact(event_id) AS distinct_event_ids
            FROM usage_events
            WHERE account_id = {account_id:String} AND ts >= now() - INTERVAL 30 DAY
            GROUP BY day ORDER BY day`,
    query_params: { account_id },
    format: "JSONEachRow",
  });
  return result.json();
};

const issueCredit = async (input: { account_id: string; amount_cents: number; reason: string }) => {
  const { rows } = await db.query(
    "INSERT INTO credits (account_id, amount_cents, reason) VALUES ($1, $2, $3) RETURNING id",
    [input.account_id, input.amount_cents, input.reason],
  );
  return { credit_id: rows[0].id, amount_cents: input.amount_cents };
};

const tools: Record<string, (input: any) => Promise<unknown>> = {
  get_account: getAccount,
  get_usage: getUsage,
  issue_credit: issueCredit,
};

const handleTicket = async (ticket: string): Promise<string> => {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: ticket }];
  for (let turn = 0; turn < 10; turn++) {
    const response = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 2048,
      system: SYSTEM_PROMPT,
      tools: TOOLS,
      messages,
    });
    messages.push({ role: "assistant", content: response.content });
    if (response.stop_reason !== "tool_use") {
      return response.content.map((b) => (b.type === "text" ? b.text : "")).join("");
    }
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const block of response.content) {
      if (block.type !== "tool_use") continue;
      const output = await observe(
        { name: block.name, spanType: "TOOL" },
        (input) => tools[block.name](input),
        block.input,
      );
      results.push({ type: "tool_result", tool_use_id: block.id, content: JSON.stringify(output) });
    }
    messages.push({ role: "user", content: results });
  }
  return "Sorry, I could not finish handling this ticket.";
};

const { ticket } = JSON.parse(readFileSync(process.argv[2], "utf8"));
try {
  console.log(await observe({ name: "handle_ticket" }, handleTicket, ticket));
} finally {
  await Promise.all([db.end(), clickhouse.close()]);
  await Laminar.shutdown();
}
