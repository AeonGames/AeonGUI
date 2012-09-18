#ifndef CLOSE_H
#define CLOSE_H
#ifdef __cplusplus
#include "Integer.h"
extern "C" {
#else
#include <stdint.h>
#endif
extern char close_name[];
extern uint32_t  close_width;
extern uint32_t  close_height;
extern uint32_t  close_bytesperpixel;
extern uint32_t  close_datasize;
extern uint8_t   close_data[];
#ifdef __cplusplus
}
#endif
#endif
