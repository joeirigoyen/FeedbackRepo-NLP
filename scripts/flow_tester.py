import json
import logging
import warnings
from pathlib import Path
from utils.tools import default_path


# Ignore warnings from output
warnings.filterwarnings("ignore")
# Set logging path
CWD = Path("./scripts")
LOG_PATH = CWD.joinpath("testing", "flowTesterLogs.log")
# Set logging parameters
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] @ %(name)s | [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH)
    ]
)
# Set logger
logger = logging.getLogger(__name__)


class FlowTester:
    @staticmethod
    def validate_config_info(config: dict) -> None:
        logger.debug("Validating configuration parameters")
        # Check if every task contains all necessary parameters to run
        tasks = config.get("tasks")
        logger.debug(f"Found {len(tasks)} tasks")
        if tasks:
            req_info = {"title", "description", "subtasks"}
            for index, task in enumerate(tasks):
                if not req_info.issubset(set(task.keys())):
                    logger.error(f"Task {index + 1} does not contain the required parameters")
                    logger.error(f"Current task's parameters are: {set(task.keys())}\n")
                    raise ValueError("One or more tasks do not contain the required parameters")
                logger.debug(f"Task {index + 1} contains all required parameters")
            logger.debug("Finished validating configuration parameters\n")
        else:
            logger.error("No attribute named \"tasks\" in configuration file\n", exc_info=True)
            raise ValueError("Could not find \"tasks\" attribute in configuration file")

    @staticmethod
    def read_config_info(config_filepath: Path) -> dict | None:
        logger.debug(f"Reading configuration JSON file at {config_filepath}")
        # Read configuration file and parse into dictionary
        with open(config_filepath, 'r') as config_file:
            try:
                config = json.load(config_file)
            except Exception as e:
                logger.error("Incorrect formatting in JSON file\n")
                return None
            else:
                logger.debug("Configuration file parsing successful\n")
                return config

    def __init__(self, config_filepath: str | Path) -> None:
        logger.debug("Creating FlowTester instance")
        # Set configuration parameters
        self.__config_filepath = default_path(config_filepath)
        self.__config = FlowTester.read_config_info(self.__config_filepath)
        if self.__config:
            try:
                FlowTester.validate_config_info(self.__config)
            except ValueError:
                logger.error("Configuration file incomplete\n", exc_info=True)
        else:
            logger.error("Could not parse JSON file correctly\n", exc_info=True)

    def run_tasks(self) -> None:
        # Get tasks from configuration dictionary
        config_tasks = self.__config["tasks"]
        logger.debug("Attempting to run tasks...")
        # Run each task
        for task in config_tasks:
            # Run subtasks from task
            subtasks = task.get('subtasks')
            logger.debug(f"Current task running: {task.get('title')}")
            logger.debug(f"Task description: {task.get('description')}")
            for subtask in subtasks:
                logger.debug(f"Running subtask: {subtask.get('name')}")
                # Prepare subtask running format
                import_path = f"{subtask.get('parent')}"
                cmd = f"{import_path}.{subtask.get('funcname')}({''.join(subtask.get('args'))})"
                # Execute subtask
                logger.debug(f"Using command: {cmd}")
                eval(f"exec(\"import {import_path}\")")
                eval(cmd)
                logger.debug(f"Finished running task: {subtask.get('name')}\n")


def run():
    CONFIG_FILENAME = "flowTesterConfig.json"
    CONFIG_FILEPATH = CWD.joinpath("testing", CONFIG_FILENAME)
    tester = FlowTester(CONFIG_FILEPATH)
    tester.run_tasks()


run()