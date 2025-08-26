import logging

def get_logger():
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.INFO
    )
    return logging.getLogger("molopt")
