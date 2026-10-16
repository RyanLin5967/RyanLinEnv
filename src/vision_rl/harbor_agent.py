"""Harbor adapter for the bounded CPU training controller."""

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class TrainingAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "vision-rl-training"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        result = await environment.exec(
            command="vision-rl train-agent --train /app/data/train.jsonl "
            "--validation /app/data/validation.jsonl --output /logs/artifacts/training",
            timeout_sec=240,
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "training.log").write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code != 0:
            raise RuntimeError(f"training controller failed with status {result.return_code}")


class VLMTrainingAgent(BaseAgent):
    """Execute a generated Harbor task's bounded GRPO training command."""

    @staticmethod
    def name() -> str:
        return "vision-rl-vlm-training"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        result = await environment.exec(command="bash /app/train.sh", timeout_sec=3500)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "training.log").write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code != 0:
            raise RuntimeError(f"VLM training failed with status {result.return_code}")


class CountTrainingAgent(BaseAgent):
    """Run the offline real-image counting task's bounded training script."""

    @staticmethod
    def name() -> str:
        return "vision-rl-count-training"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        result = await environment.exec(command="bash /app/train.sh", timeout_sec=1150)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "training.log").write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code != 0:
            raise RuntimeError(f"Count policy training failed with status {result.return_code}")
