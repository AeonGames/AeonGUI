#ifndef CLOSE_DOWN_H
#define CLOSE_DOWN_H
#ifdef __cplusplus
#include "Integer.h"
extern "C" {
#else
#include <stdint.h>
#endif
extern char close_down_name[];
extern uint32_t  close_down_width;
extern uint32_t  close_down_height;
extern uint32_t  close_down_bytesperpixel;
extern uint32_t  close_down_datasize;
extern uint8_t   close_down_data[];
#ifdef __cplusplus
}
#endif
#endif
