OpenSHMEM Functionality and Features
====================================

All OpenSHMEM-1.3 functionality is supported.  The library reports
OpenSHMEM specification version 1.5 (SHMEM_MAJOR_VERSION and
SHMEM_MINOR_VERSION, shmem_info_get_version).

Static storage is symmetric only when it is backed by writable loadable
segments of the main executable.  Unrelated mappings, including private
anonymous mappings, are not static symmetric memory.

OpenSHMEM initialization now validates all peer memheap layouts before
using remote-key state.  Incompatible layouts abort initialization with a
diagnostic.
