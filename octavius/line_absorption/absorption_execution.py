"""

Functions to handle executing the line absorption stage.

Currently under construction!!

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import OctaviusConfig, SimulationData

# internal imports
from ..log import get_logger

logger = get_logger()


def run_absorption(simulation_data: SimulationData, config: OctaviusConfig) -> None:
    """
    Executor for absorption line analysis.
    """
    if "galaxies" not in simulation_data.groups:
        logger.info("Skipping absorption line analysis: no galaxies found.")
        return
    logger.info("Preparing absorption line analysis.")
