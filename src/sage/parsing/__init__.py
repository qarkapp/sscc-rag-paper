"""Document parsing into ordered sections.

A :class:`~sage.core.protocols.Parser` turns raw bytes into ``DocumentSection``s.
The default :class:`TextParser` handles UTF-8 text, markdown, and source code; PDF
and office formats plug in behind the same protocol.
"""

from sage.parsing.text import TextParser

__all__ = ["TextParser"]
