// Recent incidents from the status service, so the agent can tell a customer
// when a bill was affected by a problem on our side.
import { observe } from "@lmnr-ai/lmnr";

export type Incident = {
  id: string;
  title: string;
  started_at: string;
  resolved_at: string | null;
  summary: string;
};

// The span name is also the mock key: an eval can answer this function
// without the status service running (mocks = {"incidents.list_recent": ...}).
export const listRecentIncidents = async ({
  days,
}: {
  days: number;
}): Promise<Incident[]> =>
  observe(
    { name: "incidents.list_recent", spanType: "TOOL" },
    async (sinceDays: number) => {
      const url = `${process.env.INCIDENTS_API_URL}/v1/incidents?since_days=${sinceDays}`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`incidents API answered ${response.status}`);
      }
      return (await response.json()) as Incident[];
    },
    days,
  );
