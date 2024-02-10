#ifdef HASDLA
#include "cuda_runtime.h"
#include "cudla.h"
#endif

extern "C" {
#ifdef HASDLA
    uint64_t getNbDLACores() {
        uint64_t nbDLA;
        cudlaDeviceGetCount(&nbDLA);
        return nbDLA;
    }
#else
    int getNbDLACores() {
        return 0;
    }
#endif
}
