// The Slack side of the agent: find the customer's newest unanswered message
// in the support channel, and reply in its thread.
import { WebClient } from "@slack/web-api";

// SLACK_API_URL is unset in production (slack.com); evals point it at a
// simulator.
const slack = new WebClient(process.env.SLACK_BOT_TOKEN, {
  slackApiUrl: process.env.SLACK_API_URL,
});

export type Question = { ts: string; ticket: string };

export const newestUnanswered = async (
  channel: string,
): Promise<Question | undefined> => {
  const { messages = [] } = await slack.conversations.history({ channel });
  // Newest first; a message without replies from a person, not a bot.
  const message = messages.find(
    (m) => m.user && !m.bot_id && !m.reply_count && !m.thread_ts,
  );
  if (!message?.ts || !message.user) return undefined;
  const { user } = await slack.users.info({ user: message.user });
  const from = `${user?.real_name ?? user?.name} <${user?.profile?.email}>`;
  return { ts: message.ts, ticket: `From: ${from}\n\n${message.text}` };
};

export const replyInThread = async (
  channel: string,
  threadTs: string,
  text: string,
) => {
  await slack.chat.postMessage({ channel, thread_ts: threadTs, text });
};
