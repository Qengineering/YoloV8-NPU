// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include "rk_common.h"

#include <iostream>
#include <fstream>
#include <cstdlib>                  // for malloc and free

/*-------------------------------------------
                  Functions
-------------------------------------------*/

void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("\tindex=%d, name=%s, \n\t\tn_dims=%d, dims=[%d, %d, %d, %d], \n\t\tn_elems=%d, size=%d, fmt=%s, \n\t\ttype=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// Function to read binary file into a buffer allocated in memory
unsigned char* load_model(const char* filename, int& fileSize)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate); // Open file in binary mode and seek to the end

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }

    fileSize = (int) file.tellg(); // Get the file size
    file.seekg(0, std::ios::beg); // Seek back to the beginning

    char* buffer = (char*)malloc(fileSize); // Allocate memory for the buffer

    if (!buffer) {
        std::cerr << "Memory allocation failed." << std::endl;
        return nullptr;
    }

    file.read(buffer, fileSize); // Read the entire file into the buffer
    file.close(); // Close the file

    return (unsigned char*) buffer;
}

