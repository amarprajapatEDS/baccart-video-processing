"""Exception types for the ingestion layer."""


class UnrecoverableSourceError(Exception):
    """The source cannot be opened and retrying will not help.

    Examples: local file that does not exist, source path is a directory,
    unsupported scheme. The watchdog catches this and stops, rather than
    backing off and reconnecting forever.
    """
