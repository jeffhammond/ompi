/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* NP: 2
 *
 * The library must report OpenSHMEM 1.5, both through the header macros
 * and through shmem_info_get_version, and the deprecated aliases must agree.
 */

#include <stdio.h>
#include <string.h>
#include <shmem.h>

#if SHMEM_MAJOR_VERSION != 1 || SHMEM_MINOR_VERSION != 5
#error "shmem.h does not report OpenSHMEM 1.5"
#endif

int main(void)
{
    int major = -1, minor = -1, fails = 0;
    char name[SHMEM_MAX_NAME_LEN];

    shmem_init();

    shmem_info_get_version(&major, &minor);
    if (1 != major || 5 != minor) {
        printf("PE %d: shmem_info_get_version returned %d.%d, expected 1.5\n",
               shmem_my_pe(), major, minor);
        fails++;
    }
    if (_SHMEM_MAJOR_VERSION != SHMEM_MAJOR_VERSION
        || _SHMEM_MINOR_VERSION != SHMEM_MINOR_VERSION
        || _SHMEM_MAX_NAME_LEN != SHMEM_MAX_NAME_LEN) {
        printf("PE %d: _SHMEM_* aliases disagree\n", shmem_my_pe());
        fails++;
    }
    if (SHMEM_MAJOR_VERSION != major || SHMEM_MINOR_VERSION != minor) {
        printf("PE %d: header macros disagree with shmem_info_get_version\n",
               shmem_my_pe());
        fails++;
    }

    memset(name, 0, sizeof(name));
    shmem_info_get_name(name);
    if ('\0' == name[0] || strlen(name) >= SHMEM_MAX_NAME_LEN) {
        printf("PE %d: bad vendor name \"%s\"\n", shmem_my_pe(), name);
        fails++;
    }

    shmem_finalize();
    return fails ? 1 : 0;
}
