from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class SupportAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "support"

    async def install(self, environment: BaseEnvironment) -> None:
        pass  # everything the agent needs is already in the image

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        # "anthropic/claude-sonnet-5" -> "claude-sonnet-5"
        model = (self.model_name or "").split("/", 1)[-1]
        await self.exec_as_agent(
            environment,
            command="python /opt/harness/run_case.py",
            env={"AGENT_MODEL": model},
            timeout_sec=240,
        )
