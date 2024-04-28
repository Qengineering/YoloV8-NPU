#ifndef _RKNN_MODEL_ZOO_COMMON_H_
#define _RKNN_MODEL_ZOO_COMMON_H_

#include "rknn_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

inline int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

void dump_tensor_attr(rknn_tensor_attr* attr);
unsigned char* load_model(const char* filename, int& fileSize);

#endif //_RKNN_MODEL_ZOO_COMMON_H_
