// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Modified by Q-engineering 4-6-2026
//

/*-------------------------------------------
                Includes
-------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cstdlib>                  // for malloc and free
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "postprocess.h"
#include "rk_common.h"
#include "rknn_api.h"

/*-------------------------------------------
                  COCO labels
-------------------------------------------*/

static const char* labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
    float f;
    float FPS[16];
    int   i, Fcnt=0;
    std::chrono::steady_clock::time_point Tbegin, Tend;
    char*          model_path = NULL;
    const char*    imagepath = argv[1];
    int            img_width          = 0;
    int            img_height         = 0;
    const float    nms_threshold      = NMS_THRESH;
    const float    box_conf_threshold = BOX_THRESH;
    int            ret;

    for(i=0;i<16;i++) FPS[i]=0.0;

    if (argc < 3) {
        fprintf(stderr,"Usage: %s [imagepath] [model]\n", argv[0]);
        return -1;
    }
    model_path = argv[2];

    printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

    // fire up the neural network
    printf("Loading mode...\n");

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    rknn_context ctx = 0;

    // Create the neural network
    int            model_data_size = 0;
    unsigned char* model_data      = load_model(model_path, model_data_size);

    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    //no need to hold the model in the buffer. Get some memory back
    free(model_data);

    if(ret < 0){
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("\nmodel input num: %d\n", io_num.n_input);
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for(uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("\nmodel output num: %d\n", io_num.n_output);
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for(uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    rknn_app_ctx.rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    {
        rknn_app_ctx.is_quant = true;
    }
    else
    {
        rknn_app_ctx.is_quant = false;
    }

    rknn_app_ctx.io_num = io_num;
    rknn_app_ctx.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW){
        printf("model input is NCHW\n");
        rknn_app_ctx.model_channel = input_attrs[0].dims[1];
        rknn_app_ctx.model_height = input_attrs[0].dims[2];
        rknn_app_ctx.model_width = input_attrs[0].dims[3];
    }
    else{
        printf("model input is NHWC\n");
        rknn_app_ctx.model_height = input_attrs[0].dims[1];
        rknn_app_ctx.model_width = input_attrs[0].dims[2];
        rknn_app_ctx.model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           rknn_app_ctx.model_height, rknn_app_ctx.model_width, rknn_app_ctx.model_channel);

    if(ret != 0){
        printf("init_yolox_model fail! ret=%d model_path=%s\n", ret, model_path);
        return ret;
    }

    // You may not need resize when src resolution equals to dst resolution
    void* buf = nullptr;
    cv::Mat orig_img;
    cv::Mat img;
    cv::Mat resized_img;
    int width   = rknn_app_ctx.model_width;
    int height  = rknn_app_ctx.model_height;
    rknn_input inputs[rknn_app_ctx.io_num.n_input];
    rknn_output outputs[rknn_app_ctx.io_num.n_output];

    printf("Start grabbing, press ESC on Live window to terminated...\n");
    while(1){
        //load image or frame
        orig_img=cv::imread(imagepath, 1);
        if(orig_img.empty()) {
            printf("Error grabbing\n");
            break;
        }

        Tbegin = std::chrono::steady_clock::now();

        //transform BGR -> RGB
        cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
        img_width  = img.cols;
        img_height = img.rows;
        //check sizes
        if (img_width != width || img_height != height) {
            cv::resize(img,resized_img,cv::Size(width,height));
            buf = (void*)resized_img.data;
        } else {
            buf = (void*)img.data;
        }
        //declare your list
        object_detect_result_list od_results;

        // Set Input Data
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].size = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
        inputs[0].buf = buf;

        // allocate inputs
        ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
        if(ret < 0){
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
        // allocate outputs
        memset(outputs, 0, sizeof(outputs));
        for(uint32_t i = 0; i < rknn_app_ctx.io_num.n_output; i++){
            outputs[i].index = i;
            outputs[i].want_float = (!rknn_app_ctx.is_quant);
        }
        // run
        rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);

        // post process
        float scale_w = (float)width / img_width;
        float scale_h = (float)height / img_height;

        post_process(&rknn_app_ctx, outputs, box_conf_threshold, nms_threshold, scale_w, scale_h, &od_results);

        // Draw Objects
        char text[256];
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det_result = &(od_results.results[i]);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            cv::rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2),cv::Scalar(255, 0, 0));

//            printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
//                   det_result->box.right, det_result->box.bottom, det_result->prop);

            //put some text
            sprintf(text, "%s %.1f%%", labels[det_result->cls_id], det_result->prop * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = det_result->box.left;
            int y = det_result->box.top - label_size.height - baseLine;
            if (y < 0) y = 0;
            if (x + label_size.width > orig_img.cols) x = orig_img.cols - label_size.width;

            cv::rectangle(orig_img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

            cv::putText(orig_img, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

        Tend = std::chrono::steady_clock::now();
        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(orig_img, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

        //show output
//        std::cout << "FPS" << f/16 << std::endl;
        imshow("Radxa zero 3W - 1,8 GHz - 4 Mb RAM", orig_img);
        char esc = cv::waitKey(5);
        if(esc == 27) break;

//      imwrite("./out.jpg", orig_img);

    }

    if(rknn_app_ctx.input_attrs  != NULL) free(rknn_app_ctx.input_attrs);
    if(rknn_app_ctx.output_attrs != NULL) free(rknn_app_ctx.output_attrs);
    if (rknn_app_ctx.rknn_ctx != 0) rknn_destroy(rknn_app_ctx.rknn_ctx);

    return 0;
}
