
#include <cstring>
#include "Controller.h"
#include "STONNEModel.h"
#include <math.h>
#include "testbench.h"
#include "types.h"
#include <vector>
#include <algorithm>
#include "MYPOOL.h"
#include "RequestPackage.h"

Controller::Controller(Config stonne_cfg, std::vector<layer_topology> layers){
    this->stonne_cfg = stonne_cfg;
    this->layers = layers;
    this->n_cycles = 0;
    this->n_conv = 0;
    this->n_pooling = 0;
    this->n_fc = 0;

    // this->write_flag = false;

    this->storage_type = stonne_cfg.storage_type;
    std::cout<<"this->storage_type : "<<this->storage_type<<std::endl;

    this->Timestamp = this->stonne_cfg.Timestamp;
    this->pooling_enabled = false;

    this->dram_read_nums = 0;
    this->dram_read_input_nums = 0;
    this->dram_read_weight_nums = 0;

    this->dram_write_nums = 0;
    this->dram_write_output_nums = 0;
    this->dram_write_neuron_state_nums = 0;

    // // DRAM中各类参数存储的空间
    // this->input_dram_size = this->stonne_cfg.m_DRAMCfg.input_dram_size;
    // this->weight_dram_size = this->stonne_cfg.m_DRAMCfg.weight_dram_size;
    // this->output_dram_size = this->stonne_cfg.m_DRAMCfg.output_dram_size;

    this->weight_width = stonne_cfg.weight_width;
    this->max_weight = stonne_cfg.max_weight;
    this->min_weight = stonne_cfg.min_weight;

    // DRAM中一次存放：输入、权重、输出和神经元状态
    this->input_offset = 0x0000; // 输入数据在DRAM中的起始地址 （单位是字节）
    this->weight_offset = 2*1024*1024;
    this->output_offset = (2 + 64) * 1024 *1024;
    this->neuron_state_offset = (2 + 64 + 2) * 1024 * 1024;
    // this->output_offset = (2 + 3) * 1024 *1024;
    // this->neuron_state_offset = (2 + 3 + 2) * 1024 * 1024;
    this->addr_offset = 64; // 一次可以从DRAM中读取8个字节的数据

    // 实例化数组当作片上buffer
    this->input_buffer_size = this->stonne_cfg.m_BufferCfg.input_buffer_size;
    this->weight_buffer_size = this->stonne_cfg.m_BufferCfg.weight_buffer_size;
    this->output_buffer_size = this->stonne_cfg.m_BufferCfg.output_buffer_size;
    this->neuron_state_buffer_size = this->stonne_cfg.m_BufferCfg.neuron_state_buffer_size;

    // std::cout<<"weight_buffer_size : "<<this->weight_buffer_size<<std::endl;

    // 片上buffer能够存储的各种数据的个数，但实际不需要存储这么多
    this->neuron_state_width = static_cast<int>(std::ceil(std::log2(this->stonne_cfg.V_th)));
    this->num_input = this->input_buffer_size*1024*8;  // 存储输入脉冲的个数
    this->num_weight = this->weight_buffer_size*1024*8/this->weight_width;  // 存储权重的个数
    this->num_output = this->output_buffer_size*1024*8;  // 存储输入脉冲的个数
    this->num_neuron_state = this->neuron_state_buffer_size*1024*8/this->neuron_state_width;  // 存储膜电位的个数

    // std::cout<<"this->num_weight : "<<this->num_weight<<std::endl;

    // this->input_buffer = new int[num_input];
    // this->weight_buffer = new int[num_weight];
    // this->output_buffer = new int[num_output];
    // this->neuron_state_buffer = new int[num_neuron_state];

    this->read_request_fifo = new Fifo(1);
    this->write_request_fifo = new Fifo(1);

}

Controller::~Controller(){
    delete read_request_fifo;
    delete write_request_fifo;
}

int Controller::completed_reads = 0;
int Controller::completed_writes = 0;

// 加载全连接层的输入和权重
int Controller::load_input_data_fc(int* ifmap, Dram* dram_instance, int num_input_obtained, int num_input_data){
    // num_input_obtained : 已经取过的数据
    // num_input_data ： 本次要取的数据
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次读取数据地址的增量
    // std::cout<<"bit_num : "<<bit_num<<std::endl;
    // std::cout<<"this->addr_offset : "<<this->addr_offset<<std::endl;

    int remain_data = num_input_obtained % bit_num; // 因为地址对齐要往前多取的数据
    int num_data = num_input_data;

    int num_input_read_request = std::ceil((remain_data+num_data) / (float)bit_num);
    int num_input_read_request_copy = num_input_read_request;
    int addr = this->input_offset + num_input_obtained/bit_num *(bit_num/8);

    // std::cout<<"addr : "<<addr<<std::endl;
    // std::cout<<"num_input_read_request : "<<num_input_read_request<<std::endl;

    while(true){
        if(num_input_read_request != 0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  // 读请求
            //std::cout<<"read weight addr is : "<<addr<<std::endl;
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset;
            num_input_read_request--;

            this->dram_read_input_nums++;  // 记录对DRAM的访问次数 
            this->dram_read_nums++;
        }

        if(completed_reads == num_input_read_request_copy){  // 读请求处理完毕
            completed_reads = 0;
            // 模拟将片外数据写入片上buffer
            for(int i=0; i<num_input_data; i++){
                //this->input_buffer[i] = ifmap[num_input_obtained+i];  // 权重数据的读取顺序在DRAM中是连续读取的
                this->ppbuf_input->next_buffer[i] = ifmap[num_input_obtained+i];
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}  

int Controller::load_weight_data_fc(int* filter, Dram* dram_instance, int i, layer_topology layer_parameters){
    // i : 第几次取权重数据
    // std::cout<<"call load_weight_data_fc and i = "<<i<<std::endl;
    // std::cout<<"filter : "<<filter<<std::endl;
    // std::cout<<"dram_instance : "<<dram_instance<<std::endl;
    // std::cout<<"layer_parameters : "<<layer_parameters.input_neuron<<std::endl;
    
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次读取数据地址的增量
    // std::cout<<"bit_num : "<<bit_num<<std::endl;
    // std::cout<<"this->addr_offset : "<<this->addr_offset<<std::endl;

    // 输入输出层神经元个数
    int input_layer = layer_parameters.input_neuron;
    int output_layer = layer_parameters.output_neuron;

    int filter_size = output_layer * input_layer;

    int num_weight_obtained = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows*input_layer;  // 已经取出的数据个数

    int start_row = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int end_row = std::min<int>(start_row+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
    int rows = end_row - start_row;

    int remain_data = (num_weight_obtained*this->weight_width) % bit_num;  
    int num_weight_data = rows * input_layer;
    int num_data = num_weight_data * this->weight_width;
    // std::cout<<"remain_data : "<<remain_data<<std::endl;
    // std::cout<<"num_data : "<<num_data<<std::endl;

    // std::cout<<"num_weight_obtained : "<<num_weight_obtained<<std::endl;
    // std::cout<<"num_weight_data : "<<num_weight_data<<std::endl;

    int num_weight_read_request = std::ceil((remain_data+num_data) / (float)bit_num);
    int num_weight_read_request_copy = num_weight_read_request;    
    // std::cout<<"num_weight_read_request : "<<num_weight_read_request<<std::endl;

    int addr = this->weight_offset + (num_weight_obtained*this->weight_width)/bit_num * (bit_num/8);

    // std::cout<<"num_weight_obtained : "<<num_weight_obtained<<std::endl;
    // std::cout<<"num_weight_read_request : "<<num_weight_read_request<<std::endl;
    // std::cout<<"addr : "<<addr<<std::endl;

    // for(int j=0; j<num_weight_data; j++){
    //     std::cout<<"j = "<<j<<std::endl;
    //     std::cout<<"num_weight_obtained+j = "<<num_weight_obtained+j<<std::endl;
    //     assert(num_weight_obtained+j>=0 && num_weight_obtained+j<filter_size);
    //     std::cout<<"filter[num_weight_obtained+j] : "<<filter[num_weight_obtained+j]<<std::endl;
    //     std::cout<<"this->ppbuf_weight->next_buffer[j] : "<<this->ppbuf_weight->next_buffer[j]<<std::endl;
    //     // this->ppbuf_weight->next_buffer[j] = filter[num_weight_obtained+j];
    // }
    
    while(true){  // 发送DRAM内存事务请求
        if(num_weight_read_request!=0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset; 
            num_weight_read_request--;
            this->dram_read_nums++;
            this->dram_read_weight_nums++;
        }

        if(completed_reads == num_weight_read_request_copy){  // 内存读取事务响应完毕
            // 将片外的数据加载到片上buffer
            // 权重数据
            for(int j=0; j<num_weight_data; j++){
                assert(num_weight_obtained + j >= 0 && num_weight_obtained + j < filter_size);
                this->ppbuf_weight->next_buffer[j] = filter[num_weight_obtained+j];
            }
            completed_reads = 0;
            //std::cout<<"All read requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
            break;
        }
        dram_instance->run();
        local_cycles++;
    }
    return local_cycles;
}

// 加载卷积层的权重数据
int Controller::load_weight_data_ppbuffer(int* filter, Dram* dram_instance, int num_weight_obtained, int num_weight_data){
    // num_weight_obtained ： 已经取出的数据数，用于计算读取地址
    // num_weight_data : 要加载的数据个数
    //std::cout<<"weight start_addr is : "<<this->weight_offset<<std::endl;
    int local_cycles = 0;

    // 进行地址对齐，计算要往前多取出的数据个数
    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次读取数据地址的增量
    // std::cout<<"bit_num : "<<bit_num<<std::endl;
    // std::cout<<"this->addr_offset : "<<this->addr_offset<<std::endl;
    int remain_data = (num_weight_obtained*this->weight_width) % bit_num;   // 计算要往前多取出的数据个数
    // std::cout<<"remain_data : "<<remain_data<<std::endl;
    int num_data = num_weight_data*this->weight_width;  // 本次取数据要取的数据量
    // std::cout<<"num_data : "<<num_data<<std::endl;
    int num_weight_read_request = std::ceil((remain_data+num_data) / (float)bit_num);  // 需要发送请求的次数
    // std::cout<<"num_weight_read_request : "<<num_weight_read_request<<std::endl;

    int num_weight_read_request_copy = num_weight_read_request;
    int addr = this->weight_offset + (num_weight_obtained*this->weight_width)/bit_num * (bit_num/8);  // 字节地址，是64的整数倍
    while(true){
        if(num_weight_read_request!=0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  // 读请求
            //std::cout<<"read weight addr is : "<<addr<<std::endl;
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset;
            num_weight_read_request--;
            
            this->dram_read_nums++;
            this->dram_read_weight_nums++;
        } 

        if(completed_reads == num_weight_read_request_copy){  // 读请求处理完毕
            completed_reads = 0;
            // 模拟将片外数据写入片上buffer
            for(int j=0; j<num_weight_data; j++){
                this->ppbuf_weight->next_buffer[j] = filter[num_weight_obtained+j];  // 权重数据的读取顺序在DRAM中是连续读取的
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}


int Controller::load_input_data_CHW(int layer_id, int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters){
    // j : 第几行卷积
    //std::cout<<" call load_input_data_CHW and    j = "<<j<<std::endl;
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次读取数据地址的增量
    // std::cout<<"bit_num : "<<bit_num<<std::endl;
    // std::cout<<"this->addr_offset : "<<this->addr_offset<<std::endl;

    int num_rows = this->records[j].num_rows; // 取多少行数据
    int start_rows = this->records[j].start_rows; //起始行
    int add_0_above = this->records[j].add_0_above;
    int add_0_below = this->records[j].add_0_below;

    // std::cout<<"start_rows : "<<start_rows<<std::endl;
    // std::cout<<"num_rows : "<<num_rows<<std::endl;
    // std::cout<<"add_0_above : "<<add_0_above<<std::endl;
    // std::cout<<"add_0_below : "<<add_0_below<<std::endl;

    int X = layer_parameters.X;
    int Y = layer_parameters.Y;

    int Y_padded = layer_parameters.Y + 2*layer_parameters.P;

    int input_buffer_size = layer_parameters.C*layer_parameters.R*Y_padded;
    int ifmap_size = layer_parameters.C * layer_parameters.X * layer_parameters.Y;

    // 一次取R*C*Y个输入数据，遍历每一个输出通道，因为每个输出通道内的数据是连续存储的
    int num_total_read_request = 0;
    bool only=true;
    while(true){
        while(only){
            for(int c=0; c<layer_parameters.C; c++){  // 遍历每个输入通道
                //std::cout<<"c: "<<c<<std::endl;

                int remain_data = (c*X*Y + start_rows*Y) % bit_num;  // 因为数据对齐，要往前多读的数据个数
                int num_data = num_rows*Y;  // 本次要取的数据个数

                int current_num_read_request = std::ceil((remain_data+num_data) / (float)bit_num);
                //std::cout<<"remain_data : "<<remain_data<<std::endl;
                //std::cout<<"num_data : "<<num_data<<std::endl;
                //std::cout<<"current_num_read_request : "<<current_num_read_request<<std::endl;

                num_total_read_request += current_num_read_request;
                
                int addr = this->input_offset + (c*X*Y + start_rows*Y)/bit_num * (bit_num/8);
                //std::cout<<"addr : "<<addr<<std::endl;
                while(current_num_read_request!=0){
                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  
                    this->read_request_fifo->push(read_request);

                    this->dram_read_input_nums++; 
                    this->dram_read_nums++;

                    current_num_read_request--;
                    addr += this->addr_offset;
                    
                    dram_instance->run();
                    local_cycles++;
                }

                // 模拟从DRAM中读取数据到片上buffer
                for(int num=0; num<add_0_above*Y_padded; num++){
                    this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + num] = 0;
                    assert((c*layer_parameters.R*Y_padded + num) < input_buffer_size);
                }

                for(int r=0; r<num_rows; r++){  // 遍历每一行，在每一行两端加P个0，中间是Y个真实数据
                    for(int num=0; num<layer_parameters.P; num++){
                        this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + num] = 0;
                        assert((c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + num) < input_buffer_size);
                    }
                    for(int num=0; num<layer_parameters.Y; num++){
                        this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num] = ifmap[c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num]; 
                        assert((c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num)>=0 && (c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num) < ifmap_size);
                        assert((c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num)>=0 && (c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num) < input_buffer_size);
                        // std::cout<<"load data from in DRAM addr "<<(c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num)<<"  to input_buffer in addr "<<(c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num)<<std::endl;
                    }
                    for(int num=0; num<layer_parameters.P; num++){
                        this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + layer_parameters.Y + num] = 0;
                        assert((c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + layer_parameters.Y + num) < input_buffer_size);
                    }
                }

                for(int num=0; num<add_0_below*Y_padded; num++){
                    this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + (add_0_above+num_rows)*Y_padded + num] = 0;
                    assert((c*layer_parameters.R*Y_padded + (add_0_above+num_rows)*Y_padded + num) < input_buffer_size);
                }
            }
            only = false;

        }

        if(completed_reads == num_total_read_request) {
            completed_reads = 0; 
            break;
        }

        // this->dram_instance->run_1(layer_id, j, completed_reads);
        this->dram_instance->run();

        local_cycles++;
    }

    //std::cout<<"num_total_read_request : "<<num_total_read_request<<std::endl;
    //std::cout<<std::endl;

    return local_cycles;
}

int Controller::load_input_data_HWC(int layer_id, int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters){

    //std::cout<<" call load_input_data_HWC and    j = "<<j<<std::endl;
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次读取数据地址的增量

    int num_rows = this->records[j].num_rows; // 取多少行数据
    int start_rows = this->records[j].start_rows; //起始行
    int add_0_above = this->records[j].add_0_above;
    int add_0_below = this->records[j].add_0_below;

    // std::cout<<"start_rows : "<<start_rows<<std::endl;
    // std::cout<<"num_rows : "<<num_rows<<std::endl;
    // std::cout<<"add_0_above : "<<add_0_above<<std::endl;
    // std::cout<<"add_0_below : "<<add_0_below<<std::endl;

    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int C = layer_parameters.C;
    int P = layer_parameters.P;
    int Y_padded = layer_parameters.Y + 2*layer_parameters.P;

    int num_data = num_rows*Y*C;  // 要从DRAM中读取多少输入数据
    int remain_data = (start_rows*Y*C)%bit_num;  // 因为地址对齐要往前多读的数据
    int num_read_input_request = std::ceil((num_data+remain_data)/(float)bit_num); // 需要发送的请求个数
    int num_read_input_request_copy = num_read_input_request;
    //std::cout<<"num_read_input_request : "<<num_read_input_request<<std::endl;

    int addr = this->input_offset + (start_rows*Y*C)/bit_num * (bit_num/8); // 起始地址
    while(true){
        if(num_read_input_request != 0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  // 读请求
            //std::cout<<"read weight addr is : "<<addr<<std::endl;
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset;
            num_read_input_request--;
            
            this->dram_read_nums++;
            this->dram_read_input_nums++;
        } 

        if(completed_reads == num_read_input_request_copy){  // 读请求处理完毕
            completed_reads = 0;
            // 模拟将片外数据写入片上buffer
            // 考虑池化
            for(int num=0; num<add_0_above*Y_padded*C; num++){
                this->ppbuf_input->next_buffer[num] = 0;
            }
            
            for(int r=0; r<num_rows; r++){  // 遍历每一个平面，加padding
                for(int num=0; num<C*P; num++){  // 如果有加padding的话，在左侧和右侧加0
                    this->ppbuf_input->next_buffer[(add_0_above+r)*C*Y_padded + num] = 0;
                    this->ppbuf_input->next_buffer[(add_0_above+r)*C*Y_padded + (P+Y)*C + num] = 0;
                }
                for(int num=0; num<C*Y; num++){
                    this->ppbuf_input->next_buffer[(add_0_above+r)*C*Y_padded + C*P + num] = ifmap[(start_rows+r)*Y*C + num];
                    //std::cout<<"ifmap addr : "<<((start_rows+r)*Y*C + num)<<"        write to input_buffer addr : "<<(add_0_above+r)*C*Y_padded + C + num<<std::endl;
                }
            }

            for(int num=0; num<add_0_below*Y_padded*C; num++){
                this->ppbuf_input->next_buffer[(add_0_above+num_rows)*Y_padded*C + num] = 0;
            }

            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

int Controller::load_input_data_HCW(int layer_id, int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters){
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次读取数据地址的增量

    int num_rows = this->records[j].num_rows; // 取多少行数据
    int start_rows = this->records[j].start_rows; //起始行
    int add_0_above = this->records[j].add_0_above;
    int add_0_below = this->records[j].add_0_below;

    // std::cout<<"start_rows : "<<start_rows<<std::endl;
    // std::cout<<"num_rows : "<<num_rows<<std::endl;
    // std::cout<<"add_0_above : "<<add_0_above<<std::endl;
    // std::cout<<"add_0_below : "<<add_0_below<<std::endl;

    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int C = layer_parameters.C;
    int P = layer_parameters.P;
    int Y_padded = layer_parameters.Y + 2*layer_parameters.P;

    int num_data = num_rows*Y*C;  // 要从DRAM中读取多少输入数据
    int remain_data = (start_rows*Y*C)%bit_num;  // 因为地址对齐要往前多读的数据
    int num_read_input_request = std::ceil((num_data+remain_data)/(float)bit_num); // 需要发送的请求个数
    int num_read_input_request_copy = num_read_input_request;
    // std::cout<<"num_read_input_request : "<<num_read_input_request<<std::endl;

    int addr = this->input_offset + (start_rows*Y*C)/bit_num * (bit_num/8); // 起始地址
    while(true){
        if(num_read_input_request != 0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  // 读请求
            //std::cout<<"read weight addr is : "<<addr<<std::endl;
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset;
            num_read_input_request--;
            
            this->dram_read_nums++;
            this->dram_read_input_nums++;
        } 

        if(completed_reads == num_read_input_request_copy){  // 读请求处理完毕
            completed_reads = 0;
            // 模拟将片外数据写入片上buffer
            // 考虑池化
            for(int num=0; num<add_0_above*Y_padded*C; num++){
                this->ppbuf_input->next_buffer[num] = 0;
            }
            
            for(int r=0; r<num_rows; r++){  // 遍历每一个平面，加padding
                for(int channels=0; channels<C; channels++){  // 遍历每一个通道，每一个通道内的数据是一行数据，是连续存储的，在其前后加padding
                    for(int num=0; num<P; num++){
                        this->ppbuf_input->next_buffer[(add_0_above+r)*C*Y_padded + channels*Y_padded + num] = 0;
                        this->ppbuf_input->next_buffer[(add_0_above+r)*C*Y_padded + channels*Y_padded + P + Y + num] = 0;
                        // std::cout<<"local : ["<<((add_0_above+r)*C*Y_padded + channels*Y_padded + num)<<"]  set 0"<<std::endl;
                        // std::cout<<"local : ["<<((add_0_above+r)*C*Y_padded + channels*Y_padded + P + Y + num)<<"]  set 0"<<std::endl;
                    }

                    for(int num=0; num<Y; num++){
                        this->ppbuf_input->next_buffer[(add_0_above+r)*C*Y_padded + channels*Y_padded + P + num] = ifmap[(start_rows+r)*Y*C + channels*Y + num];
                        // std::cout<<"ifmap local ["<<((start_rows+r)*Y*C + channels*Y + num)<<"]  to input_buffer local ["<<((add_0_above+r)*C*Y_padded + channels*Y_padded + 1 + num)<<"]"<<std::endl;
                    }
                }
            }

            for(int num=0; num<add_0_below*Y_padded*C; num++){
                this->ppbuf_input->next_buffer[(add_0_above+num_rows)*Y_padded*C + num] = 0;
            }

            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

int Controller::load_input_data_HCW_bank(int layer_id, int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters){
    // std::cout<<"-------------- call load_input_data_HCW_bank ----------------"<<std::endl;
    // std::cout<<"                   j : "<<j<<std::endl;
    
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次读取数据地址的增量

    // int num_rows = this->records[j].num_rows; // 取多少行数据
    // int start_rows = this->records[j].start_rows; //起始行
    // int add_0_above = this->records[j].add_0_above;
    // int add_0_below = this->records[j].add_0_below;

    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int C = layer_parameters.C;
    int R = layer_parameters.R;
    int P = layer_parameters.P;
    int stride = layer_parameters.stride;
    int Y_padded = layer_parameters.Y + 2*layer_parameters.P;
    int X_ = (X + 2*P - R)/stride + 1;

    assert(P==1); // 目前只支持P==1的池化

    int num_data;
    int remain_data = 0;
    int num_read_input_request;
    int num_read_input_request_copy;
    int addr;
    int rows_index;
    if(j==0){ // 第一次取数据，取(R-P)行
        num_data = (R-P)*C*Y; // 要从DRAM中取多少数据
        num_read_input_request = std::ceil(num_data/(float)bit_num);
        num_read_input_request_copy = num_read_input_request;
        addr = this->input_offset;
        // std::cout<<"num_read_input_request : "<<num_read_input_request<<std::endl;
        // std::cout<<"addr : "<<addr<<std::endl;
    } else if(j==X_-1 && P>0) { // 最后一次取数据，如果P>0，不用取数据，直接补0
        for(int num=0; num<Y_padded*C; num++){
            this->ppbuf_bank->next_buffer[num] = 0;
        }
        return 0;
    } else {  // 取一行数据到this->ppbuf_bank->next_buffer;
        rows_index = R-P-1+j;  // 取数据的行索引
        // std::cout<<"rows_index : "<<rows_index<<std::endl;
        remain_data = (rows_index*C*Y)%bit_num;
        num_data = C*Y;
        num_read_input_request = std::ceil((remain_data+num_data)/(float)bit_num);
        num_read_input_request_copy = num_read_input_request;
        addr = this->input_offset + (rows_index*C*Y)/bit_num*(bit_num/8);
    }

    while(true){
        if(num_read_input_request != 0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  // 读请求
            //std::cout<<"read weight addr is : "<<addr<<std::endl;
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset;
            num_read_input_request--;
            
            this->dram_read_nums++;
            this->dram_read_input_nums++;
        }

        if(completed_reads == num_read_input_request_copy){  // 读请求处理完毕
            completed_reads = 0;

            if(j==0){
                for(int num=0; num<C*Y_padded; num++){
                    this->input_base_bank_0[num] = 0;
                }
                for(int channels=0; channels<C; channels++){
                    this->input_base_bank_1[channels*Y_padded] = 0;
                    this->ppbuf_bank->next_buffer[channels*Y_padded] = 0;
                    this->input_base_bank_1[channels*Y_padded + P + Y] = 0;
                    this->ppbuf_bank->next_buffer[channels*Y_padded + P + Y] = 0;

                    for(int num=0; num<Y; num++){
                        this->input_base_bank_1[channels*Y_padded + P + num] = ifmap[channels*Y+num]; // 第一行数据
                        this->ppbuf_bank->next_buffer[channels*Y_padded + P + num] = ifmap[C*Y + channels*Y + num]; // 第二行数据
                    }
                }
            } else {  // 只取一行数据（一个平面）
                for(int channels=0; channels<C; channels++){
                    this->ppbuf_bank->next_buffer[channels*Y_padded] = 0;
                    this->ppbuf_bank->next_buffer[channels*Y_padded + P + Y] = 0;
                    for(int num=0; num<Y; num++){
                        this->ppbuf_bank->next_buffer[channels*Y_padded + P + num] = ifmap[rows_index*Y*C + channels*Y +num];
                    }
                }
            }

            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    // std::cout<<"below is input_buffer data : "<<std::endl;
    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     std::cout<<"bank1:"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         //std::cout<<"num : "<<num<<std::endl;
    //         std::cout<<this->input_base_bank_0[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    //     std::cout<<"bank2"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         std::cout<<this->input_base_bank_1[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    //     std::cout<<"bank3"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         std::cout<<this->ppbuf_bank->next_buffer[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    return local_cycles;
}


int Controller::store_neuron_state_CHW(int* nfmap, Dram* dram_instance, int i, int cols, layer_topology layer_parameters){
    // i : 权重循环
    // cols : 输出通道数
    int local_cycles = 0;

    // 进行地址对齐
    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    int num_neuron_state = X_*Y_*cols * this->neuron_state_width;
    int remain_neuron_state_data = (i*X_*Y_*this->stonne_cfg.m_MSNetworkCfg.ms_cols*this->neuron_state_width) % bit_num;
    int num_neuron_state_write_request = std::ceil((remain_neuron_state_data + num_neuron_state) / (float)bit_num);
    int num_neuron_state_write_request_copy = num_neuron_state_write_request;
    //std::cout<<"num_neuron_state_write_request : "<<num_neuron_state_write_request<<std::endl;

    int addr_neuron_state = this->neuron_state_offset +  (i*X_*Y_*this->stonne_cfg.m_MSNetworkCfg.ms_cols*this->neuron_state_width)/bit_num * (bit_num/8);
    while(true){
        if(num_neuron_state_write_request!=0){
            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr_neuron_state,true);
            this->write_request_fifo->push(write_request);
            addr_neuron_state += this->addr_offset;
            num_neuron_state_write_request--;

            this->dram_write_nums++;
            this->dram_write_neuron_state_nums++;
        }

        if(completed_writes == num_neuron_state_write_request_copy) {
            completed_writes = 0; 
            for(int num=0; num<num_neuron_state/this->neuron_state_width; num++){
                nfmap[(i*X_*Y_*this->stonne_cfg.m_MSNetworkCfg.ms_cols) + num] = this->neuron_state_buffer[num];
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
} 

int Controller::store_neuron_state_HWC(int* nfmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters){

    // std::cout<<"************** call store_neuron_state_HWC ***********"<<std::endl;
    // std::cout<<"this->neuron_state_buffer[2] : "<<this->neuron_state_buffer[2]<<std::endl;

    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int K = layer_parameters.K;
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    
    int num_total_write_request = 0;
    bool only = true;

    while(true){
        while(only){
            for(int p=0; p<Y_; p++){
                // int remain_data = (j*K*Y_ + i*this->stonne_cfg.m_MSNetworkCfg.ms_cols*Y_ + p*K)*this->neuron_state_width % bit_num;
                int remain_data = (j*K*Y_ + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols)*this->neuron_state_width % bit_num;
                int num_neuron_state = cols*this->neuron_state_width;
                int num_neuron_state_write_request = std::ceil((remain_data+num_neuron_state) / (float)bit_num);
                num_total_write_request += num_neuron_state_write_request;

                int addr = this->neuron_state_offset + (j*K*Y_ + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols)*this->neuron_state_width/bit_num * (bit_num/8);
                while(num_neuron_state_write_request!=0){
                    std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr,true);
                    this->write_request_fifo->push(write_request);
                    addr += this->addr_offset;
                    num_neuron_state_write_request--;

                    this->dram_instance->run();
                    local_cycles++;

                    this->dram_write_nums++;
                    this->dram_write_neuron_state_nums++;
                }

                // 模拟将片上buffer的数据写入DRAM
                for(int q=0; q<cols; q++){
                    // nfmap[(j*K*Y_ + i*this->stonne_cfg.m_MSNetworkCfg.ms_cols*Y_ + p*K) + q] = this->neuron_state_buffer[p*cols + q];
                    nfmap[(j*K*Y_ + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols) + q] = this->neuron_state_buffer[p*cols + q];
                }
            }

            only = false;
        }

        if(completed_writes == num_total_write_request){
            completed_writes = 0;
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }

    // 中间状态数据写入DRAM之后，重置为0
    int neuron_state_buffer_size = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    std::memset(this->neuron_state_buffer, 0, sizeof(int) * neuron_state_buffer_size);
    std::memset(this->neuron_state_buffer_cpu, 0, sizeof(int) * neuron_state_buffer_size);

    return local_cycles;
}

int Controller::store_neuron_state_HCW(int* nfmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters){
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int K = layer_parameters.K;
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    int num_neuron_state = Y_*cols*this->neuron_state_width;
    int remain_data = (j*Y_*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_ )*this->neuron_state_width % bit_num;
    int num_neuron_state_write_request = std::ceil((remain_data+num_neuron_state)/(float)bit_num);
    int num_neuron_state_write_request_copy = num_neuron_state_write_request;

    int addr = this->neuron_state_offset + (j*Y_*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_ )*this->neuron_state_width/bit_num * (bit_num/8);

    while(true){
        if(num_neuron_state_write_request != 0){
            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr,true);
            this->write_request_fifo->push(write_request);
            addr += this->addr_offset;
            num_neuron_state_write_request--;

            this->dram_write_nums++;
            this->dram_write_neuron_state_nums++;
        }

        if(completed_writes == num_neuron_state_write_request_copy) {
            completed_writes = 0; 
            // 模拟将片上buffer数据写入DRAM 
            for(int num=0; num<num_neuron_state/this->neuron_state_width; num++){
                nfmap[(j*Y_*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_) + num] = this->neuron_state_buffer[num];
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    // // 中间状态数据写入DRAM之后，重置为0
    // int neuron_state_buffer_size = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    // std::memset(this->neuron_state_buffer, 0, sizeof(int) * neuron_state_buffer_size);
    // std::memset(this->neuron_state_buffer_cpu, 0, sizeof(int) * neuron_state_buffer_size);

    return local_cycles;
}


int Controller::store_pooling_output_CHW(int* ofmap, Dram* dram_instance, int i, int cols, layer_topology layer_parameters){
    // X_pool = (j-1)/2  当前输出行（池化结果）在输出特征图的第几行
    int local_cycles = 0;
    
    // 进行地址对齐
    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    int num_output = X_*Y_/4*cols;
    int remain_output_data = (i*X_*Y_/4*this->stonne_cfg.m_MSNetworkCfg.ms_cols) % bit_num;
    int num_output_write_requset = std::ceil((remain_output_data + num_output) / (float)bit_num);
    int num_output_write_request_copy = num_output_write_requset;
    // std::cout<<"num_output_write_requset : "<<num_output_write_requset<<std::endl;

    int addr_output = this->output_offset + (i*X_*Y_/4*this->stonne_cfg.m_MSNetworkCfg.ms_cols)/bit_num * (bit_num/8);
    while(true){
        if(num_output_write_requset!=0){
            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr_output,true);
            this->write_request_fifo->push(write_request);
            addr_output += this->addr_offset;
            num_output_write_requset--;

            this->dram_write_nums++;
            this->dram_write_output_nums++;
        }

        if(completed_writes == num_output_write_request_copy) {
            completed_writes = 0; 
            // 模拟将片上buffer数据写入DRAM 
            for(int num=0; num<num_output; num++){
                ofmap[(i*X_*Y_/4*this->stonne_cfg.m_MSNetworkCfg.ms_cols) + num] = this->output_buffer[num];
            }

            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

int Controller::store_pooling_output_HWC(int* ofmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters){
    //std::cout<<"call store_pooling_output_HWC-----------------------------"<<std::endl;
    int local_cycles = 0;
    // 进行地址对齐
    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int K = layer_parameters.K;
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    j = (j-1)/2;  // 该输出行在池化后的特征图中的位置

    // std::cout<<"j : "<<j<<std::endl;

    int num_total_write_request = 0;
    bool only = true;
    while(true){
        while(only){
            for(int p=0; p<Y_/2; p++){
                // std::cout<<"p : "<<p<<std::endl;
                int remain_data = (j*K*Y_/2 + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols) % bit_num;
                int num_output = cols;
                int num_output_write_request = std::ceil((remain_data+num_output) / (float)bit_num);
                // std::cout<<"num_output_write_request : "<<num_output_write_request<<std::endl;
                num_total_write_request += num_output_write_request;

                int addr = this->output_offset + (j*K*Y_/2 + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols)/bit_num * (bit_num/8);
                while(num_output_write_request != 0){
                    std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr,true);
                    this->write_request_fifo->push(write_request);
                    addr += this->addr_offset;
                    num_output_write_request--;

                    this->dram_instance->run();
                    local_cycles++;

                    this->dram_write_nums++;
                    this->dram_write_output_nums++;
                }

                // 模拟将片上buffer的数据写入DRAM
                for(int q=0; q<cols; q++){
                    ofmap[(j*K*Y_/2 + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols)  + q] = this->output_buffer[p*cols + q];
                    // std::cout<<"output buffer addr : "<<(p*cols + q)<<"     write to ofmap addr "<<((j*K*Y_/2 + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols)  + q)<<std::endl;
                }

            }
            only = false;
        }

        if(completed_writes == num_total_write_request){
            completed_writes = 0;
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }

    // std::cout<<"num_total_write_request : "<<num_total_write_request<<std::endl;

    // int output_buffer_size = Y_/2 * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    // std::memset(this->output_buffer, 0, sizeof(int) * output_buffer_size);
    // std::memset(this->output_buffer_cpu, 0, sizeof(int) * output_buffer_size);

    return local_cycles;
}

int Controller::store_pooling_output_HCW(int* ofmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters){
    //std::cout<<"call store_pooling_output_HCW-----------------------------"<<std::endl;
    int local_cycles = 0;

    // 进行地址对齐
    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int K = layer_parameters.K;
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    j = (j-1)/2;  // 该输出行在池化后的特征图中的位置

    int num_output = Y_/2*cols; // 输出buffer中所有的数据
    int remain_data = (j*Y_/2*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_/2)%bit_num;
    int num_output_write_requset = std::ceil((num_output+remain_data)/(float)bit_num);
    int num_output_write_request_copy = num_output_write_requset;
    // std::cout<<"num_output_write_requset : "<<num_output_write_requset<<std::endl;

    int addr = this->output_offset + (j*Y_/2*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_/2)/bit_num * (bit_num/8);
    while(true){
        if(num_output_write_requset != 0){
            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr,true);
            this->write_request_fifo->push(write_request);
            addr += this->addr_offset;
            num_output_write_requset--;

            this->dram_write_nums++;
            this->dram_write_output_nums++;
        }

        if(completed_writes == num_output_write_request_copy) {
            completed_writes = 0; 
            // 模拟将片上buffer数据写入DRAM 
            for(int num=0; num<num_output; num++){
                ofmap[(j*Y_/2*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_/2) + num] = this->output_buffer[num];
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    int output_buffer_size = Y_/2 * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    std::memset(this->output_buffer, 0, sizeof(int) * output_buffer_size);
    std::memset(this->output_buffer_cpu, 0, sizeof(int) * output_buffer_size);
    return local_cycles;
}


int Controller::store_conv_output_CHW(int* ofmap, Dram* dram_instance, int i, int cols, layer_topology layer_parameters){
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    int num_output = X_*Y_*cols;
    int remain_output_data = (i*X_*Y_*this->stonne_cfg.m_MSNetworkCfg.ms_cols) % bit_num;
    int num_output_write_requset = std::ceil((remain_output_data + num_output) / (float)bit_num);
    int num_output_write_request_copy = num_output_write_requset;
    //std::cout<<"num_output_write_requset : "<<num_output_write_requset<<std::endl;

    int addr_output = this->output_offset + (i*X_*Y_*this->stonne_cfg.m_MSNetworkCfg.ms_cols)/bit_num * (bit_num/8);

    // 写输出数据
    while(true){
        if(num_output_write_requset!=0){
            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr_output,true);
            this->write_request_fifo->push(write_request);
            addr_output += this->addr_offset;
            num_output_write_requset--;

            this->dram_write_nums++;
            this->dram_write_output_nums++;
        }

        if(completed_writes == num_output_write_request_copy) {
            completed_writes = 0; 
            // 模拟将片上buffer数据写入DRAM 
            for(int num=0; num<num_output; num++){
                ofmap[(i*X_*Y_*this->stonne_cfg.m_MSNetworkCfg.ms_cols) + num] = this->output_buffer[num];
            }

            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

int Controller::store_conv_output_HWC(int* ofmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters){
    // i : 权重循环
    // j ： 输入数据循环
    // cols : 输出通道数
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int K = layer_parameters.K;
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    
    int num_total_write_request = 0;
    bool only = true;
    while(true){
        while(only){
            for(int p=0; p<Y_; p++){  // 遍历输出buffer的每一行，其中每一行是各个通道内相同位置的数据，在DRAM中是连续存储的
                int remain_data = (j*K*Y_ + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols) % bit_num;
                int num_output = cols;
                int num_output_write_request = std::ceil((remain_data+num_output) / (float)bit_num);
                num_total_write_request += num_output_write_request;

                int addr = this->output_offset + (j*K*Y_ + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols) /bit_num * (bit_num/8);
                while(num_output_write_request != 0){
                    std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr,true);
                    this->write_request_fifo->push(write_request);
                    addr += this->addr_offset;
                    num_output_write_request--;

                    this->dram_instance->run();
                    local_cycles++;

                    this->dram_write_nums++;
                    this->dram_write_output_nums++;
                }

                // 模拟将片上buffer的数据写入DRAM
                for(int q=0; q<cols; q++){
                    ofmap[(j*K*Y_ + p*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols)  + q] = this->output_buffer[p*cols + q];
                }
            }
            only = false;
        }

        if(completed_writes == num_total_write_request){
            completed_writes = 0;
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }

    int output_buffer_size = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    std::memset(this->output_buffer, 0, sizeof(int) * output_buffer_size);
    std::memset(this->output_buffer_cpu, 0, sizeof(int) * output_buffer_size);
    
    return local_cycles;
}

int Controller::store_conv_output_HCW(int* ofmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters) {
    // 整行的数据在DRAM中是连续存储的
    int local_cycles = 0;

    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int K = layer_parameters.K;
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    int num_output = Y_*cols; // 输出buffer中所有的数据
    int remain_data = (j*Y_*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_ )%bit_num;
    int num_output_write_requset = std::ceil((num_output+remain_data)/(float)bit_num);
    int num_output_write_request_copy = num_output_write_requset;
    //std::cout<<"num_output_write_requset : "<<num_output_write_requset<<std::endl;

    int addr = this->output_offset + (j*Y_*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_)/bit_num * (bit_num/8);
    while(true){
        if(num_output_write_requset != 0){
            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr,true);
            this->write_request_fifo->push(write_request);
            addr += this->addr_offset;
            num_output_write_requset--;

            this->dram_write_nums++;
            this->dram_write_output_nums++;
        }

        if(completed_writes == num_output_write_request_copy) {
            completed_writes = 0; 
            // 模拟将片上buffer数据写入DRAM 
            for(int num=0; num<num_output; num++){
                ofmap[(j*Y_*K + i*stonne_cfg.m_MSNetworkCfg.ms_cols*Y_) + num] = this->output_buffer[num];
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    // int output_buffer_size = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    // std::memset(this->output_buffer, 0, sizeof(int) * output_buffer_size);
    // std::memset(this->output_buffer_cpu, 0, sizeof(int) * output_buffer_size);

    return local_cycles;
}


int Controller::store_output_and_neuronstate_data_fc(int* ofmap, int* nfmap, Dram* dram_instance, int num_output_obtained, int num_output_write_once, layer_topology layer_parameters){
    // i : 权重循环

    // std::cout<<"--------------------- call store_output_and_neuronstate_data_fc ------------------------"<<std::endl;
    // std::cout<<"this->output_buffer[0] : "<<this->output_buffer[0]<<std::endl;
    // std::cout<<"num_output_obtained : "<<num_output_obtained<<std::endl;
    // std::cout<<"num_output_write_once : "<<num_output_write_once<<std::endl;
    int local_cycles = 0;

    // 进行地址对齐
    int bit_num = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // DRAM一次读取返回的数据量，单位bit
    this->addr_offset = bit_num / 8; // 每次写数据，地址的增量

    int num_output = num_output_write_once;
    int num_neuron_state = num_output_write_once * this->neuron_state_width;

    int num_output_write_request = std::ceil(num_output/(float)bit_num);
    int num_neuron_state_write_request = std::ceil(num_neuron_state/(float)bit_num);

    // std::cout<<"num_output_write_request : "<<num_output_write_request<<std::endl;
    // std::cout<<"num_neuron_state_write_request : "<<num_neuron_state_write_request<<std::endl;

    int num_output_write_request_copy = num_output_write_request;
    int num_neuron_state_write_request_copy = num_neuron_state_write_request;    

    int addr_output = this->output_offset + num_output_obtained/bit_num * (bit_num/8);
    int addr_neuron_state = this->neuron_state_offset + num_output_obtained*this->neuron_state_width/bit_num * (bit_num/8);

    while(true){
        if(num_neuron_state_write_request!=0){
            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr_neuron_state,true);
            this->write_request_fifo->push(write_request);
            addr_neuron_state += this->addr_offset;
            num_neuron_state_write_request--;

            this->dram_write_nums++;
            this->dram_write_neuron_state_nums++;
        }

        if(completed_writes == num_neuron_state_write_request_copy) {
            completed_writes = 0; 
            for(int num=0; num<num_output_write_once; num++){
                nfmap[num_output_obtained + num] = this->neuron_state_buffer[num];
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    while(true){
        if(num_output_write_request!=0){
            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(addr_output,true);
            this->write_request_fifo->push(write_request);
            addr_output += this->addr_offset;
            num_output_write_request--;

            this->dram_write_nums++;
            this->dram_write_output_nums++;
        }

        if(completed_writes == num_output_write_request_copy) {
            completed_writes = 0; 
            for(int num=0; num<num_output_write_once; num++){
                ofmap[num_output_obtained + num] = this->output_buffer[num];
            }
            // std::cout<<"ofmap[0] : "<<ofmap[0]<<std::endl;
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    // int output_layer = layer_parameters.output_neuron;

    // int start = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // int end = std::min<int>(start+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
    // int num = end - start;

    // int num_output_write_request = std::ceil(num / (float)dram_instance->dram->GetBusBits());
    // int num_neuron_state_write_request = std::ceil(num*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
    // int num_total_write_request = num_output_write_request + num_neuron_state_write_request;
    // int num_total_write_request_copy = num_total_write_request;

    // int addr_output = this->output_offset + start/8;
    // int addr_neuron_state = this->neuron_state_offset + start*std::ceil(std::log2(this->stonne_cfg.V_th))/8;
    // while(true){
    //     while(num_total_write_request != 0){
    //         if(num_output_write_request != 0){
    //             std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr_output,true);  // 写请求
    //             this->write_request_fifo->push(read_request);
    //             addr_output += this->addr_offset;
    //             num_output_write_request--;
    //         } else if(num_neuron_state_write_request != 0){
    //             std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr_neuron_state,true);  // 写请求
    //             this->write_request_fifo->push(read_request);
    //             addr_neuron_state += this->addr_offset;
    //             num_neuron_state_write_request--;
    //         }
    //         num_total_write_request--;
    //         this->dram_instance->run();
    //         local_cycles++;
    //     }

    //     if(completed_writes == num_total_write_request_copy) {
    //         completed_writes = 0; 
    //         break;
    //     }

    //     this->dram_instance->run();
    //     local_cycles++;
    // }

    // // 模拟将数据写入DRAM
    // for(int k=0; k<num; k++){
    //     ofmap[start + k] = this->output_buffer[k];
    //     nfmap[start + k] = this->neuron_state_buffer[k];
    // }

    // // std::cout<<"write output[0] : "<<this->output_buffer[0]<<std::endl;
    // // std::cout<<"write neuron[0] : "<<this->neuron_state_buffer[0]<<std::endl;

    return local_cycles;
}

int Controller::process_fc(int i, layer_topology layer_parameters){

    // std::cout<<"--------------------call process_fc--------------------"<<std::endl;
    // std::cout<<"         i : "<<i<<std::endl;
    int local_cycles = 0;
    
    // 输入输出层神经元个数
    int input_layer = layer_parameters.input_neuron;
    int output_layer = layer_parameters.output_neuron;
    
    int start_row = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int end_row = std::min<int>(start_row+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
    int rows = end_row - start_row;

    // int start_col = j*this->num_input;
    // int end_col = std::min<int>(start_col+this->num_input, input_layer);
    // int cols = end_col - start_col;

    // std::cout<<"------------input data is : "<<std::endl;
    // for(int num=0; num<input_layer; num++){
    //     std::cout<<this->ppbuf_input->current_buffer[num]<<"  ";
    // }

    // std::cout<<std::endl;
    // std::cout<<"------------weight data is : "<<std::endl;
    // for(int p=0; p<rows; p++){
    //     for(int q=0; q<input_layer; q++){
    //         std::cout<<this->ppbuf_weight->current_buffer[p*input_layer+q]<<"  ";
    //     }
    //     std::cout<<std::endl;
    // }



    this->output_regfile_conv = new int[rows]();
    this->output_regfile_conv_cpu = new int[rows]();
    this->neuron_state_regfile_conv = new int[rows]();
    this->neuron_state_regfile_conv_cpu = new int[rows]();

    Stonne* stonne_instance = new Stonne(stonne_cfg);
    // matrixMultiply(rows,cols,1, this->weight_buffer, this->input_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
    matrixMultiply(rows,input_layer,1, this->ppbuf_weight->current_buffer, this->ppbuf_input->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
    
    // stonne_instance->loadDenseGEMM(layer_name, 1, cols, rows, this->weight_buffer, this->input_buffer, this->output_buffer, this->neuron_state_buffer, CNN_DATAFLOW);
    stonne_instance->loadDenseGEMM(layer_name, 1, input_layer, rows, this->ppbuf_weight->current_buffer, this->ppbuf_input->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
    //std::cout<<"debug"<<std::endl;
    stonne_instance->loadGEMMTile(1, 1,rows);
    stonne_instance->run();

    local_cycles += stonne_instance->n_cycles;

    // 对当前tile的计算结果进行验证
    for(int num=0; num<rows; num++){
        float difference = fabs(this->output_regfile_conv[num] - this->output_regfile_conv_cpu[num]) + fabs(this->neuron_state_regfile_conv[num] - this->neuron_state_regfile_conv_cpu[num]);
        if(difference>0){
            std::cout << "ERROR position " << num <<  ": Value ofmap simulator: " << this->output_regfile_conv[num] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[num] << std::endl;
            std::cout << "ERROR position " << num <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[num] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[num] << std::endl;
            std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
            assert(false);
        }
    }
    // std::cout<<"this->output_regfile_conv[0] : "<<this->output_regfile_conv[0]<<std::endl;
    // std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;

    // 将寄存器文件中的输出结果累积到输出buffer
    for(int num=0; num<rows; num++){
        // assert((i*this->stonne_cfg.m_MSNetworkCfg.ms_rows + num) < 512);
        // assert((i*this->stonne_cfg.m_MSNetworkCfg.ms_rows + num) < 512);
        this->output_buffer[(i*this->stonne_cfg.m_MSNetworkCfg.ms_rows)%512 + num] = this->output_regfile_conv[num];
        this->neuron_state_buffer[(i*this->stonne_cfg.m_MSNetworkCfg.ms_rows)%512 + num] = this->neuron_state_regfile_conv[num];
    }

    //std::cout<<"debug 1"<<std::endl;
    delete stonne_instance;
    //std::cout<<"debug 2"<<std::endl;
    delete[] this->output_regfile_conv;
    //std::cout<<"debug 3"<<std::endl;
    delete[] this->output_regfile_conv_cpu;
    //std::cout<<"debug 4"<<std::endl;
    delete[] this->neuron_state_regfile_conv;
    //std::cout<<"debug 5"<<std::endl;
    delete[] this->neuron_state_regfile_conv_cpu;
    //std::cout<<"debug 6"<<std::endl;

    return local_cycles;
}


int Controller::im2col_CHW(int start, int num, layer_topology layer_parameters){
    // start : 起始的窗口
    // num ： 一共有多少个窗口
    // std::cout<<"start : "<<start<<std::endl;
    // std::cout<<"num : "<<num<<std::endl;
    // std::cout<<"The begin sorted data is : "<<std::endl;
    // for(int p=0; p<this->bankSize; p++){
    //     std::cout<<this->im2col_bank[p]<<"  ";
    // }
    // std::cout<<"--------------------------"<<std::endl;

    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    // int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // this->input_arranged_buffer = new int[input_arranged_buffer_size];
    // std::cout<<"input buffer : "<<this->ppbuf_input->current_buffer[Y_padded+1]<<std::endl;
    int flag = 0;
    if(start==0){
        //std::cout<<"debug"<<std::endl;
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                for(int q=0; q<S; q++){
                    this->im2col_bank[c*R*S + r*S + q] = this->ppbuf_input->current_buffer[c*R*Y_padded + r*Y_padded + q];
                    local_cycles++;
                }
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<im2col_bank[p]<<" ";
        // }
        // std::cout<<std::endl;

        //this->input_arranged.push_back(im2col_bank);

        // 将排序好的数据写入待计算buffer
        for(int i=0; i<this->bankSize; i++){
            this->ppbuf_input_arranged->next_buffer[i] = this->im2col_bank[i];
        }
        flag = 1;
        num--;  // 从窗口个数 -1
        start++;  // 起始窗口 +1
    }

    for(int i=0; i<num; i++){ 
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                // 移位
                for(int q=0; q<S-1; q++){
                    this->im2col_bank[c*R*S + r*S + q] = this->im2col_bank[c*R*S + r*S + q + 1];
                }
                // 取数据
                this->im2col_bank[c*R*S + r*S + (S-1)] = this->ppbuf_input->current_buffer[c*R*Y_padded + r*Y_padded + (S+(start+i-1))];
                local_cycles++;
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        //this->input_arranged.push_back(im2col_bank);

        for(int j=0; j<this->bankSize; j++){
            this->ppbuf_input_arranged->next_buffer[(flag+i)*this->bankSize + j] = this->im2col_bank[j];
        }
    }

    return local_cycles;
}

int Controller::im2col_HWC(int start, int num, layer_topology layer_parameters){
    // start : 起始的窗口
    // num ： 一共有多少个窗口
    // std::cout<<"========== call im2col ============"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    int flag=0;
    if(start==0){  // 第一个卷积窗口重排，特殊处理
        for(int r=0; r<R; r++){ // 
            for(int num=0; num<C*S; num++){
                this->im2col_bank[r*S*C + num] = this->ppbuf_input->current_buffer[r*C*Y_padded + num];
                local_cycles++;
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<this->im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        // 将排序好的数据写入待机算buffer
        for(int i=0; i<this->bankSize; i++){
            this->ppbuf_input_arranged->next_buffer[i] = this->im2col_bank[i];
        }
        flag = 1;
        num--;
        start++;
    }

    for(int i=0; i<num; i++){
        for(int r=0; r<R; r++){
            for(int p=0; p<S-1; p++){  // 移位部分
                for(int q=0; q<C; q++){
                    this->im2col_bank[r*S*C + p*C + q] = this->im2col_bank[r*S*C + p*C + q + C];
                }
            }
            // 取数据
            for(int q=0; q<C; q++){
                this->im2col_bank[r*S*C + (S-1)*C + q] = this->ppbuf_input->current_buffer[r*C*Y_padded + (S+(start+i)-1)*C + q];
                local_cycles++;
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<this->im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        for(int j=0; j<this->bankSize; j++){
            this->ppbuf_input_arranged->next_buffer[(flag+i)*this->bankSize + j] = this->im2col_bank[j];
        }
    }

    return local_cycles;
}

int Controller::im2col_HCW(int start, int num, layer_topology layer_parameters){
    // start : 起始的窗口
    // num ： 一共有多少个窗口
    // std::cout<<"========== call im2col_HCW ============"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    int flag = 0;
    if(start==0){
        for(int r=0; r<R; r++){
            for(int p=0; p<C; p++){
                for(int q=0; q<S; q++){
                    this->im2col_bank[r*S*C + p*S + q] = this->ppbuf_input->current_buffer[r*C*Y_padded + p*Y_padded + q];
                    local_cycles++;
                }
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<this->im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        // 将排序好的数据写入待机算buffer
        for(int i=0; i<this->bankSize; i++){
            this->ppbuf_input_arranged->next_buffer[i] = this->im2col_bank[i];
        }
        flag = 1;
        num--;
        start++;
    }

    for(int i=0; i<num; i++){
        for(int r=0; r<R; r++){
            for(int p=0; p<C; p++){
                for(int q=0; q<(S-1); q++){ // 移位
                    this->im2col_bank[r*C*S + p*S + q] = this->im2col_bank[r*C*S + p*S + q + 1];
                }   
                // 取数据
                this->im2col_bank[r*C*S + p*S + (S-1)] = this->ppbuf_input->current_buffer[r*C*Y_padded + p*Y_padded + (S+(start+i)-1)];
                local_cycles++;
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<this->im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        for(int j=0; j<this->bankSize; j++){
            this->ppbuf_input_arranged->next_buffer[(flag+i)*this->bankSize + j] = this->im2col_bank[j];
        }
    }
    
    return local_cycles;
}

int Controller::im2col_HCW_bank(int start, int num, layer_topology layer_parameters){
    // std::cout<<"========== call im2col_HCW_bank ============"<<std::endl;
    // std::cout<<"start : "<<start<<std::endl;
    // std::cout<<"num : "<<num<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    int flag = 0;
    if(start==0){
        for(int p=0; p<C; p++){
            for(int q=0; q<S; q++){
                this->im2col_bank[p*S + q] = this->input_base_bank_0[p*Y_padded + q];
                this->im2col_bank[C*S + p*S + q] = this->input_base_bank_1[p*Y_padded + q];
                this->im2col_bank[2*C*S +p*S + q] = this->ppbuf_bank->current_buffer[p*Y_padded + q];
                local_cycles++;
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<this->im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        // 将排序好的数据写入待机算buffer
        for(int i=0; i<this->bankSize; i++){
            this->ppbuf_input_arranged->next_buffer[i] = this->im2col_bank[i];
        }
        flag = 1;
        num--;
        start++;
    }

    for(int i=0; i<num; i++){
        for(int r=0; r<R; r++){
            for(int p=0; p<C; p++){
                for(int q=0; q<(S-1); q++){ // 移位
                    this->im2col_bank[r*C*S + p*S + q] = this->im2col_bank[r*C*S + p*S + q + 1];
                }   
            }
        }

        // 取数据
        for(int p=0; p<C; p++){
            this->im2col_bank[p*S + (S-1)] = this->input_base_bank_0[p*Y_padded + (S+(start+i)-1)];
            this->im2col_bank[C*S + p*S + (S-1)] = this->input_base_bank_1[p*Y_padded + (S+(start+i)-1)];
            this->im2col_bank[2*C*S + p*S + (S-1)] = this->ppbuf_bank->current_buffer[p*Y_padded + (S+(start+i)-1)];
            local_cycles++;
        }

        for(int j=0; j<this->bankSize; j++){
            this->ppbuf_input_arranged->next_buffer[(flag+i)*this->bankSize + j] = this->im2col_bank[j];
        }
    }

    return local_cycles;
}


int Controller::process_conv_CHW(int layer_id, int i, int j, int cols, layer_topology layer_parameters){
    //std::cout<<"call process_conv_CHW"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    int Y_padded = Y + 2*P;

    // std::cout<<"below is input_buffer data : "<<std::endl;
    // for(int c=0; c<C; c++){
    //     std::cout<<"channel : "<<c<<std::endl;
    //     for(int p=0; p<R; p++){
    //         for(int q=0; q<Y_padded; q++){
    //             std::cout<<this->ppbuf_input->current_buffer[c*R*Y_padded + p*Y_padded + q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // std::cout<<"debug1"<<std::endl;

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();


    // 第一次排序，将排序好的数据存在next_buffer中
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }

    // std::cout<<"debug3"<<std::endl;

    int n_cycles_im2col = im2col_CHW(0,num,layer_parameters);
    local_cycles += n_cycles_im2col;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);
    //std::cout<<"Sort first input_arranged : "<<n_cycles_im2col<<std::endl;

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        // std::cout<<"current num_tile : "<<num_tile<<"-------------"<<std::endl;
        int n_cycles_im2col;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col = im2col_CHW(start, num, layer_parameters);
            //std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        } else {
            n_cycles_im2col = 0;
            //std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current-start_current;
        

        // 调用脉动阵列，下面的计算过程和对下一个tile的数据进行排序是并行的
        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        // matrixMultiply_new(num, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->weight_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
        //std::cout<<"begin load layer"<<std::endl;
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        // stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num, this->ppbuf_input_arranged->current_buffer, this->weight_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        //std::cout<<"begin load tile"<<std::endl;
        stonne_instance->loadGEMMTile(cols,1,num_current);
        //std::cout<<"begin run"<<std::endl;
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();
        int n_cycles_compute = stonne_instance->n_cycles;  // 计算所需时间
        //std::cout<<"compute the tile need cycles : "<<n_cycles_compute<<std::endl;
        local_cycles += std::max(n_cycles_im2col, n_cycles_compute);
        // std::cout<<"current local_cycles is : "<<local_cycles<<std::endl;
        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }

    }

    // 遍历每一列，将输出结果累积到输出buffer
    for(int p=0; p<cols; p++){

        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期
        
        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            // this->output_buffer[p*Y_ + q] = packed_col_output[q];
            // this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
            this->output_buffer[X_*Y_*p + j*Y_ + q] = packed_col_output[q];
            this->neuron_state_buffer[X_*Y_*p + j*Y_ + q] = packed_col_neuron_state[q];
        }
    }
    
    delete stonne_instance;
    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    return local_cycles;
}

int Controller::process_conv_HWC(int layer_id, int i, int j, int cols, layer_topology layer_parameters) {
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    // this->output_regfile_conv = new int[Y_*cols]();
    // this->output_regfile_conv_cpu = new int[Y_*cols]();
    // this->neuron_state_regfile_conv = new int[Y_*cols]();
    // this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 第一次排序，将排序好的数据存在next_buffer中
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }
    int n_cycles_im2col = im2col_HWC(0,num,layer_parameters);
    local_cycles += n_cycles_im2col;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        // std::cout<<"---------------- num_tile -----------------------"<< num_tile<<std::endl;
        int n_cycles_im2col;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col = im2col_HWC(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        } else {
            n_cycles_im2col = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current-start_current;

        // std::cout<<"below is this->ppbuf_input_arranged->current_buffer data : "<<std::endl;
        // for(int p=0; p<num_current; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_input_arranged->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        // std::cout<<"below is this->ppbuf_weight->current_buffer data : "<<std::endl;
        // for(int p=0; p<cols; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_weight->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;


        // 调用脉动阵列，下面的计算过程和对下一个tile的数据进行排序是并行的
        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_buffer, this->neuron_state_buffer, CNN_DATAFLOW);
        stonne_instance->loadGEMMTile(cols,1,num_current);
        //std::cout<<"begin run"<<std::endl;
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();
        int n_cycles_compute = stonne_instance->n_cycles;  // 计算所需时间
        // std::cout<<"compute the tile need cycles : "<<n_cycles_compute<<std::endl;
        local_cycles += std::max(n_cycles_im2col, n_cycles_compute);
        // std::cout<<"current local_cycles is : "<<local_cycles<<std::endl;
        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }
        // std::cout<<"************************************"<<std::endl;
        // std::cout<<"this->output_buffer[1] : "<<this->output_buffer[1]<<std::endl;
        // std::cout<<"this->neuron_state_buffer[1] : "<<this->neuron_state_buffer[1]<<std::endl;
        // std::cout<<"************************************"<<std::endl;
    }


    // // 遍历每一列，将寄存器中的结果累积到输出buffer
    // for(int p=0; p<cols; p++){
    //     std::vector<int> packed_col_output(Y_,0);
    //     std::vector<int> packed_col_neuron_state(Y_,0);
    //     for(int q=0; q<Y_; q++){
    //         packed_col_output[q] = this->output_regfile_conv[q*cols+p];
    //         packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
    //     }
    //     local_cycles++; // 假设打包需要一个周期

    //     // 将打包后的数据写入片上buffer
    //     for(int q=0; q<Y_; q++){
    //         this->output_buffer[p*Y_ + q] = packed_col_output[q];
    //         this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
    //     }
    // }

    delete stonne_instance;
    // delete[] this->output_regfile_conv;
    // delete[] this->output_regfile_conv_cpu;
    // delete[] this->neuron_state_regfile_conv;
    // delete[] this->neuron_state_regfile_conv_cpu;

    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    return local_cycles;
}

int Controller::process_conv_HCW(int layer_id, int i, int j, int cols, layer_topology layer_parameters) {
    // std::cout<<"------------ call process_conv_HCW -----------"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;


    // std::cout<<"below is input_buffer data : "<<std::endl;

    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<R; p++){
    //         for(int q=0; q<Y_padded; q++){
    //             // int index = p * Y_padded * C + q * C + channels;  // HWC layout
    //             int index = p * C * Y_padded + channels * Y_padded + q; // HCW layout
    //             std::cout<<this->ppbuf_input->current_buffer[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 第一次排序，将排序好的数据存在next_buffer中
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }
    int n_cycles_im2col = im2col_HCW(0,num,layer_parameters);
    // std::cout<<"Sort first input_arranged : "<<n_cycles_im2col<<std::endl;
    local_cycles += n_cycles_im2col;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        // std::cout<<"---------------- num_tile -----------------------"<< num_tile<<std::endl;
        int n_cycles_im2col;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col = im2col_HCW(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        } else {
            n_cycles_im2col = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current-start_current;

        // std::cout<<"below is this->ppbuf_input_arranged->current_buffer data : "<<std::endl;
        // for(int p=0; p<num_current; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_input_arranged->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        // std::cout<<"below is this->ppbuf_weight->current_buffer data : "<<std::endl;
        // for(int p=0; p<cols; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_weight->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;


        // 调用脉动阵列，下面的计算过程和对下一个tile的数据进行排序是并行的
        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        stonne_instance->loadGEMMTile(cols,1,num_current);
        //std::cout<<"begin run"<<std::endl;
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();
        int n_cycles_compute = stonne_instance->n_cycles;  // 计算所需时间
        // std::cout<<"compute the tile need cycles : "<<n_cycles_compute<<std::endl;
        local_cycles += std::max(n_cycles_im2col, n_cycles_compute);
        // std::cout<<"current local_cycles is : "<<local_cycles<<std::endl;
        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }
        // std::cout<<"************************************"<<std::endl;
        // std::cout<<"this->output_buffer[1] : "<<this->output_buffer[1]<<std::endl;
        // std::cout<<"this->neuron_state_buffer[1] : "<<this->neuron_state_buffer[1]<<std::endl;
        // std::cout<<"************************************"<<std::endl;
    }

    // 遍历每一列，将寄存器文件中的内容进行转置存储在输出buffer中
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期

        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->output_buffer[p*Y_ + q] = packed_col_output[q];
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
        }
    }

    delete stonne_instance;
    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    return local_cycles;
}

int Controller::process_conv_HCW_bank(int layer_id, int i, int j, int cols, layer_topology layer_parameters){
    //std::cout<<"------------ call process_conv_HCW_bank -----------"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;


    // std::cout<<"below is input_buffer data : "<<std::endl;

    // for(int channels=0; channels<C; channels++){
    //     //std::cout<<"debug"<<std::endl;
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     //std::cout<<"bank1:"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         //std::cout<<"num : "<<num<<std::endl;
    //         std::cout<<this->input_base_bank_0[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    //     //std::cout<<"bank2"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         std::cout<<this->input_base_bank_1[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    //     //std::cout<<"bank3"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         std::cout<<this->ppbuf_bank->current_buffer[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 第一次排序，将排序好的数据存在next_buffer中
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }
    int n_cycles_im2col = im2col_HCW_bank(0,num,layer_parameters);
    // std::cout<<"Sort first input_arranged : "<<n_cycles_im2col<<std::endl;
    local_cycles += n_cycles_im2col;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        int n_cycles_im2col;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col = im2col_HCW_bank(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        } else {
            n_cycles_im2col = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current-start_current;

        // // 打印im2col排序后的数据
        // std::cout<<"below is this->ppbuf_input_arranged->current_buffer data : "<<std::endl;
        // for(int p=0; p<num_current; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         if(q>0 && q%(C*S)==0){
        //             std::cout<<"  ";
        //         }
        //         std::cout<<this->ppbuf_input_arranged->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        // std::cout<<"below is this->ppbuf_weight->current_buffer data : "<<std::endl;
        // for(int p=0; p<cols; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_weight->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        // 调用脉动阵列，下面的计算过程和对下一个tile的数据进行排序是并行的
        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        stonne_instance->loadGEMMTile(cols,1,num_current);
        //std::cout<<"begin run"<<std::endl;
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();
        int n_cycles_compute = stonne_instance->n_cycles;  // 计算所需时间
        //std::cout<<"compute the tile need cycles : "<<n_cycles_compute<<std::endl;
        local_cycles += std::max(n_cycles_im2col, n_cycles_compute);
        // std::cout<<"current local_cycles is : "<<local_cycles<<std::endl;
        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }
    }

    // 遍历每一列，将寄存器文件中的内容进行转置存储在输出buffer中
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期

        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->output_buffer[p*Y_ + q] = packed_col_output[q];
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
        }
    }

    // 该输入循环处理完成之后，将input bank的数据向前移动
    for(int num=0; num<C*Y_padded; num++){
        this->input_base_bank_0[num] = this->input_base_bank_1[num];
        this->input_base_bank_1[num] = this->ppbuf_bank->current_buffer[num];
    }

    delete stonne_instance;
    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    return local_cycles;
}


int Controller::process_conv_and_pooling_CHW(int layer_id, int i, int j, int cols, int count_rows, layer_topology layer_parameters){
    // std::cout<<"call process_conv_and_pooling"<<std::endl;
    // count_rows : 用于累积输出到片上SRAM的计数器
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    int Y_padded = Y + 2*P;

    // std::cout<<"below is input buffer data : "<<std::endl;
    // for(int channel=0; channel<C; channel++){
    //     for(int p=0; p<R; p++){
    //         for(int q=0; q<Y_padded; q++){
    //             std::cout<<this->ppbuf_input->current_buffer[channel*R*Y_padded + p*Y_padded +q]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }
    

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 第一次排序
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }

    int n_cycles_im2col_first = im2col_CHW(0,num,layer_parameters);
    local_cycles += n_cycles_im2col_first;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        // std::cout<<"num_tile : "<<num_tile<<"-------------"<<std::endl;

        int n_cycles_im2col_next;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col_next = im2col_CHW(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        } else {
            n_cycles_im2col_next = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current - start_current;
        // std::cout<<"num_current : "<<num_current<<std::endl;

        // std::cout<<"input : ------"<<std::endl;
        // for(int num=0; num<9; num++){
        //     std::cout<<this->ppbuf_input_arranged->current_buffer[num]<<"  ";
        // }
        // std::cout<<std::endl;
        // for(int num=0; num<9; num++){
        //     std::cout<<this->ppbuf_weight->current_buffer[num]<<"  ";
        // }
        // std::cout<<std::endl;

        // 调用脉动阵列
        // matrixMultiply_new(num, this->bankSize, cols, this->input_arranged_buffer, this->weight_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        // std::cout<<"cpu neuron_state[1] : "<<this->neuron_state_regfile_conv_cpu[1]<<std::endl;
        // std::cout<<"cpu output[1] : "<<this->output_regfile_conv_cpu[1]<<std::endl;
        
        //std::cout<<"begin load layer"<<std::endl;
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        // stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num, this->input_arranged_buffer, this->weight_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        //std::cout<<"begin load tile"<<std::endl;
        stonne_instance->loadGEMMTile(cols,1,num_current);
        //std::cout<<"begin run"<<std::endl;
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();
        // std::cout<<"sim neuron_state[1] : "<<this->neuron_state_regfile_conv[1]<<std::endl;
        // std::cout<<"sim output[1] : "<<this->output_regfile_conv[1]<<std::endl;

        int n_cycles_compute = stonne_instance->n_cycles;
        // std::cout<<"compute the tile need cycles : "<<n_cycles_compute<<std::endl;
        local_cycles += std::max(n_cycles_im2col_next, n_cycles_compute);
        // std::cout<<"current local_cycles is : "<<local_cycles<<std::endl;

        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }

    }

    // 对输出和神经元状态分别处理
    // 1. 神经元状态累积到output_buffer
    // 2. 输出累积到片上SRAM
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期
        //std::cout<<packed_col_neuron_state[0]<<std::endl;
        
        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->on_chip_sram[p*2*Y_ + count_rows*Y_ + q] = packed_col_output[q];
            this->neuron_state_buffer[X_*Y_*p + j*Y_ + q] = packed_col_neuron_state[q];
        }
    }

    delete stonne_instance;
    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;
    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    return local_cycles;
}

int Controller::process_conv_and_pooling_HWC(int layer_id, int i, int j, int cols, int count_rows, layer_topology layer_parameters){
    // 通道后置
    // 不需要对neuron state数据进行累积
    // std::cout<<"call process_conv_and_pooling_HWC"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    int Y_padded = Y + 2*P;

    // std::cout<<"below is input_buffer data : "<<std::endl;

    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<R; p++){
    //         for(int q=0; q<Y_padded; q++){
    //             int index = p * Y_padded * C + q * C + channels;  // HWC layout
    //             // int index = p * C * Y_padded + channels * Y_padded + q; // HCW layout
    //             std::cout<<this->ppbuf_input->current_buffer[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    // 输出数据需要进一步累积进行池化
    // 神经元状态数据在一行卷积计算完成之后，直接写入DRAM
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    // this->neuron_state_regfile_conv = new int[Y_*cols]();
    // this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 第一次排序
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }

    int n_cycles_im2col_first = im2col_HWC(0,num,layer_parameters);
    local_cycles += n_cycles_im2col_first;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        int n_cycles_im2col_next;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col_next = im2col_HWC(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        } else {
            n_cycles_im2col_next = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current - start_current;

        // std::cout<<"below is this->ppbuf_input_arranged->current_buffer data : "<<std::endl;
        // for(int p=0; p<num_current; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_input_arranged->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        // std::cout<<"below is this->ppbuf_weight->current_buffer data : "<<std::endl;
        // for(int p=0; p<cols; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_weight->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_buffer, CNN_DATAFLOW);
        stonne_instance->loadGEMMTile(cols,1,num_current);
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();

        int n_cycles_compute = stonne_instance->n_cycles;
        local_cycles += std::max(n_cycles_im2col_next, n_cycles_compute);
        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }

        //std::cout << "\033[1;31mT test passed\033[0m" << std::endl;
    }

    // std::cout<<"************************************"<<std::endl;
    // std::cout<<"this->neuron_state_buffer[2] : "<<this->neuron_state_buffer[2]<<std::endl;
    // std::cout<<"************************************"<<std::endl;

    // 对输出数据和神经元状态数据分别处理
    // 1. 神经元状态数据直接写入DRAM
    // 2. 输出累积到片上SRAM，进一步进行池化
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期

        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->on_chip_sram[p*2*Y_ + count_rows*Y_ + q] = packed_col_output[q];
        }
    }

    delete stonne_instance;
    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;

    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    return local_cycles;
}

int Controller::process_conv_and_pooling_HCW(int layer_id, int i, int j, int cols, int count_rows, layer_topology layer_parameters){
    // std::cout<<"call process_conv_and_pooling_HCW"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    int Y_padded = Y + 2*P;

    // std::cout<<"below is input_buffer data : "<<std::endl;

    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<R; p++){
    //         for(int q=0; q<Y_padded; q++){
    //             int index = p * Y_padded * C + q * C + channels;  // HWC layout
    //             // int index = p * C * Y_padded + channels * Y_padded + q; // HCW layout
    //             std::cout<<this->ppbuf_input->current_buffer[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    // 输出数据需要进一步累积进行池化
    // 神经元状态数据在一行卷积计算完成之后，直接写入DRAM
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 第一次排序
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }

    int n_cycles_im2col_first = im2col_HCW(0,num,layer_parameters);
    local_cycles += n_cycles_im2col_first;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        int n_cycles_im2col_next;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col_next = im2col_HCW(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        } else {
            n_cycles_im2col_next = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current - start_current;

        // std::cout<<"below is this->ppbuf_input_arranged->current_buffer data : "<<std::endl;
        // for(int p=0; p<num_current; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_input_arranged->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        // std::cout<<"below is this->ppbuf_weight->current_buffer data : "<<std::endl;
        // for(int p=0; p<cols; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_weight->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        stonne_instance->loadGEMMTile(cols,1,num_current);
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();

        int n_cycles_compute = stonne_instance->n_cycles;
        local_cycles += std::max(n_cycles_im2col_next, n_cycles_compute);
        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }

        //std::cout << "\033[1;31mT test passed\033[0m" << std::endl;
    }

    // 对输出数据和神经元状态数据分别处理
    // 1. 神经元状态数据转置存入片上buffer
    // 2. 输出累积到片上SRAM，进一步进行池化
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期

        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
            this->on_chip_sram[p*2*Y_ + count_rows*Y_ + q] = packed_col_output[q];
        }
    }

    delete stonne_instance;
    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    return local_cycles;
}

int Controller::process_conv_and_pooling_HCW_bank(int layer_id, int i, int j, int cols, int count_rows, layer_topology layer_parameters){
    // std::cout<<"call process_conv_and_pooling_HCW_bank"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    // std::cout<<"below is input_buffer data : "<<std::endl;

    // for(int channels=0; channels<C; channels++){
    //     //std::cout<<"debug"<<std::endl;
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     //std::cout<<"bank1:"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         //std::cout<<"num : "<<num<<std::endl;
    //         std::cout<<this->input_base_bank_0[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    //     //std::cout<<"bank2"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         std::cout<<this->input_base_bank_1[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    //     //std::cout<<"bank3"<<std::endl;
    //     for(int num=0; num<Y_padded; num++){
    //         std::cout<<this->ppbuf_bank->current_buffer[channels*Y_padded + num]<<"  ";
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    // 输出数据需要进一步累积进行池化
    // 神经元状态数据在一行卷积计算完成之后，直接写入DRAM
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 第一次排序
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }

    int n_cycles_im2col_first = im2col_HCW_bank(0,num,layer_parameters);
    local_cycles += n_cycles_im2col_first;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        int n_cycles_im2col_next;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col_next = im2col_HCW_bank(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        } else {
            n_cycles_im2col_next = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current - start_current;

        // std::cout<<"below is this->ppbuf_input_arranged->current_buffer data : "<<std::endl;
        // for(int p=0; p<num_current; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_input_arranged->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        // std::cout<<"below is this->ppbuf_weight->current_buffer data : "<<std::endl;
        // for(int p=0; p<cols; p++){
        //     for(int q=0; q<this->bankSize; q++){
        //         std::cout<<this->ppbuf_weight->current_buffer[p*this->bankSize + q]<<"  ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;

        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        stonne_instance->loadGEMMTile(cols,1,num_current);
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();

        int n_cycles_compute = stonne_instance->n_cycles;
        local_cycles += std::max(n_cycles_im2col_next, n_cycles_compute);
        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }
    }

    // 对输出数据和神经元状态数据分别处理
    // 1. 神经元状态数据转置存入片上buffer
    // 2. 输出累积到片上SRAM，进一步进行池化
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期

        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
            this->on_chip_sram[p*2*Y_ + count_rows*Y_ + q] = packed_col_output[q];
        }
    }

    // 该输入循环处理完成之后，将input bank的数据向前移动
    for(int num=0; num<C*Y_padded; num++){
        this->input_base_bank_0[num] = this->input_base_bank_1[num];
        this->input_base_bank_1[num] = this->ppbuf_bank->current_buffer[num];
    }

    delete stonne_instance;
    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    return local_cycles;
}


int Controller::process_pooling_CHW(int i, int j, int cols, layer_topology layer_parameters){
    // 执行池化，并将池化结果累积到输出buffer中
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    this->output_regfile_pooling = new int[Y_*cols/2];
    this->output_regfile_pooling_cpu = new int[Y_*cols/2];

    MYPOOL* pooling_instance = new MYPOOL(stonne_cfg);
    pooling_instance->loadPOOLLayer(Y_, cols, this->on_chip_sram, this->output_regfile_pooling);
    pooling_instance->run();
    local_cycles += pooling_instance->n_cycle;
    
    // // 验证模拟器计算的结果
    pool2x2(this->on_chip_sram, this->output_regfile_pooling_cpu, Y_, cols);

    for(int k=0; k<Y_*cols/2; k++) {
        float difference = fabs(this->output_regfile_pooling[k] - this->output_regfile_pooling_cpu[k]);
        if(difference>0){
            std::cout<<"error location : "<<k<<std::endl;
            std::cout<<"output_buffer : "<<this->output_regfile_pooling[k]<<std::endl;
            std::cout<<"output_buffer_cpu : "<<this->output_regfile_pooling_cpu[k]<<std::endl;
            assert(false);
        }
    }
    // std::cout << "\033[1;32m POOLING layer Test passed correctly\033[0m" << std::endl << std::endl;

    // 将输出结果累积到片上output buffer
    int X_pool = (j-1)/2; // 当前输出行在输出特征图中的第几行
    for(int p=0; p<cols; p++){
        for(int q=0; q<Y_/2; q++){
            this->output_buffer[X_/2*Y_/2*p + X_pool*Y_/2 + q] = this->output_regfile_pooling[p*Y_/2 + q];
        }
    }

    return local_cycles;
}

int Controller::process_pooling_HWC(int i, int j, int cols, layer_topology layer_parameters){
    // 通道后置
    // 将池化的结果转置为通道后置的形式，以便将数据写入DRAM
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    this->output_regfile_pooling = new int[Y_*cols/2];
    this->output_regfile_pooling_cpu = new int[Y_*cols/2];

    MYPOOL* pooling_instance = new MYPOOL(stonne_cfg);
    pooling_instance->loadPOOLLayer(Y_, cols, this->on_chip_sram, this->output_regfile_pooling);
    pooling_instance->run();
    local_cycles += pooling_instance->n_cycle;
    
    // // 验证模拟器计算的结果
    pool2x2(this->on_chip_sram, this->output_regfile_pooling_cpu, Y_, cols);

    for(int k=0; k<Y_*cols/2; k++) {
        float difference = fabs(this->output_regfile_pooling[k] - this->output_regfile_pooling_cpu[k]);
        if(difference>0){
            std::cout<<"error location : "<<k<<std::endl;
            std::cout<<"output_buffer : "<<this->output_regfile_pooling[k]<<std::endl;
            std::cout<<"output_buffer_cpu : "<<this->output_regfile_pooling_cpu[k]<<std::endl;
            assert(false);
        }
    }

    // 将this->output_regfile_pooling转置为通道后置的存储顺序
    for(int p=0; p<Y_/2; p++){
        std::vector<int> packed_col_output(cols,0);
        for(int q=0; q<cols; q++){
            packed_col_output[q] = this->output_regfile_pooling[q*Y_/2+p];
        }
        local_cycles++;  // 打包需要一个周期

        for(int q=0; q<cols; q++){
            this->output_buffer[p*cols + q] = packed_col_output[q];
        }
    }

    delete[] this->output_regfile_pooling;
    delete[] this->output_regfile_pooling_cpu;

    return local_cycles;
}

int Controller::process_pooling_HCW(int i, int j, int cols, layer_topology layer_parameters){
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    // this->output_regfile_pooling = new int[Y_*cols/2];
    // this->output_regfile_pooling_cpu = new int[Y_*cols/2];

    MYPOOL* pooling_instance = new MYPOOL(stonne_cfg);
    pooling_instance->loadPOOLLayer(Y_, cols, this->on_chip_sram, this->output_buffer);
    pooling_instance->run();
    local_cycles += pooling_instance->n_cycle;
    
    // // 验证模拟器计算的结果
    pool2x2(this->on_chip_sram, this->output_buffer_cpu, Y_, cols);

    for(int k=0; k<Y_*cols/2; k++) {
        float difference = fabs(this->output_buffer[k] - this->output_buffer_cpu[k]);
        if(difference>0){
            std::cout<<"error location : "<<k<<std::endl;
            std::cout<<"output_buffer : "<<this->output_buffer[k]<<std::endl;
            std::cout<<"output_buffer_cpu : "<<this->output_buffer_cpu[k]<<std::endl;
            assert(false);
        }
    }

    return local_cycles;
}


void Controller::PingPongBuffer_Init(PingPong_Buffer* ppbuf, int* buffer_0, int* buffer_1){
    // 初始化乒乓buffer
    ppbuf->current_buffer = buffer_1;
    ppbuf->next_buffer = buffer_0;
    ppbuf->buffer_toggle = true;
}

void Controller::PingPongBuffer_Switch(PingPong_Buffer* ppbuf, int* buffer_0, int* buffer_1){
    // 乒乓buffer切换
    ppbuf->buffer_toggle = !ppbuf->buffer_toggle;
    if(ppbuf->buffer_toggle) {
        ppbuf->current_buffer = buffer_1;
        ppbuf->next_buffer = buffer_0;
    } else {
        ppbuf->current_buffer = buffer_0;
        ppbuf->next_buffer = buffer_1;
    }
}

void Controller::read_callback(uint64_t addr){
    //std::cout << "Read completed at address: 0x" << std::hex << addr << std::endl;
    //std::cout << "Read completed at address: 0x" << std::hex << addr << " at cycle " << std::dec << globalCycle << std::endl;
    completed_reads++;
}

void Controller::write_callback(uint64_t addr){
    //std::cout << "Write completed at address: 0x" << std::hex << addr << std::endl;
    //std::cout << "Write completed at address: 0x" << std::hex << addr << " at cycle " << std::dec << globalCycle << std::endl;
    completed_writes++;
} 


std::tuple<int*, int*, int*, int*> Controller::runFC(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    // 全连接层控制逻辑
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                  Call runFC function                                    "<<std::endl;
    std::cout<<"                                  Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    // 输入和权重双buffer，模块化，input_buffer容量有限

    // std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m       "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_fc++;
    this->layer_name = "fc"+std::to_string(this->n_fc);

    // 提取层参数，输入层神经元个数和输出层神经元个数
    int output_layer = layer_parameters.output_neuron;
    int input_layer = layer_parameters.input_neuron;

    // 建模存储真实的数据
    int ifmap_size = input_layer;
    int filter_size = output_layer * input_layer;
    // std::cout<<"filter_size : "<<filter_size<<std::endl;

    int ofmap_size = output_layer;
    int nfmap_size = output_layer;

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim
    this->dram_instance = new Dram(read_callback, write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 例化片上输入buffer和权重buffer
    int num_input_buffer_need = input_layer;
    int num_weight_buffer_need = input_layer * this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // int num_output_buffer_need = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // int num_neuron_state_buffer_need = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    assert(num_input_buffer_need <= this->num_input);
    assert(num_weight_buffer_need <= this->num_weight);
    // assert(num_output_buffer_need <= this->num_output);
    // assert(num_neuron_state_buffer_need <= this->num_neuron_state);

    // this->input_buffer = new int[num_input_buffer_need];
    // this->weight_buffer = new int[num_weight_buffer_need];
    // this->output_buffer = new int[num_output_buffer_need]();
    // this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    // this->output_buffer_cpu = new int[num_output_buffer_need]();
    // this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    // // 累积中间结果
    // int* acc_inter_result = new int[num_neuron_state_buffer_need]();
    // int* acc_inter_output = new int[num_input_buffer_need]();

    // 输出buffer和神经元状态buffer
    int num_output_buffer_need = dram_instance->dram->GetBusBits() * dram_instance->dram->GetBurstLength();  // 512bits
    assert(num_output_buffer_need <= this->num_output);
    assert(num_output_buffer_need <= this->num_neuron_state);
    this->output_buffer = new int[num_output_buffer_need];
    this->neuron_state_buffer = new int[num_output_buffer_need];

    // 从片外读取数据的计数器
    int num_input_buffer_fold = std::ceil(input_layer / (float)this->num_input);  // 片上input_buffer容量有限，计算需要分几次读取
    assert(num_input_buffer_fold==1);  // 目前只支持片上buffer容量足够大的情况
    int num_weight_buffer_fold = std::ceil(output_layer / (float)this->stonne_cfg.m_MSNetworkCfg.ms_rows);  // 权重数据也需要分批读取，需要（num_input_buffer_fold*num_weight_buffer_fold）

    // 第一次取输入数据和权重数据
    int num_input_obtained = 0;
    int num_input_data_first = std::min<int>(this->num_input, input_layer);
    int n_cycles_load_first_input = load_input_data_fc(ifmap, this->dram_instance, num_input_obtained, num_input_data_first);
    num_input_obtained += num_input_data_first;
    this->n_cycles += n_cycles_load_first_input;
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);  // 切换缓冲区
    std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 第一次取权重数据
    int n_cycles_load_first_weight = load_weight_data_fc(filter, this->dram_instance, 0, layer_parameters);
    this->n_cycles += n_cycles_load_first_weight;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);  // 切换缓冲区
    std::cout<<"load the first weight need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    int num_output_obtained = 0;  // 计算得到的输出个数
    int num_output_write_once = 0;
    for(int i=0; i<num_weight_buffer_fold; i++){
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        int start_row = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
        int end_row = std::min<int>(start_row+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
        int rows = end_row - start_row;

        // 取下一块权重数据
        int n_cycles_load_next_weight;
        if(i+1 < num_weight_buffer_fold){
            //std::cout<<"*******************************"<<std::endl;
            n_cycles_load_next_weight = load_weight_data_fc(filter, this->dram_instance, i+1, layer_parameters);
            std::cout<<"load the next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
        } else {
            n_cycles_load_next_weight = 0;
            std::cout<<"load the next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
        }

        // 调用脉动阵列计算
        int n_cycles_process_fc = process_fc(i,layer_parameters);
        num_output_write_once += rows;
        std::cout<<"process_fc need cycles : "<<n_cycles_process_fc<<std::endl;
        this->n_cycles += std::max(n_cycles_load_next_weight, n_cycles_process_fc);
        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);
        std::cout<<"The current weight loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
        std::cout<<std::endl;

        int n_cycles_write_result;
        if(num_output_write_once >= num_output_buffer_need){  // 累积够
            n_cycles_write_result = store_output_and_neuronstate_data_fc(ofmap, nfmap, dram_instance, num_output_obtained, num_output_write_once, layer_parameters);
            num_output_obtained += num_output_write_once;
            // std::cout<<"n_cycles_write_result : "<<n_cycles_write_result<<std::endl;
            std::cout<<"write all result need cycles : "<<n_cycles_write_result<<std::endl;
            num_output_write_once = 0;

            if(i == num_weight_buffer_fold-1){
                this->n_cycles += n_cycles_write_result;
                std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
            } else {
                std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
            }
        } else {
            n_cycles_write_result = 0;
            // std::cout<<"n_cycles_write_result : "<<n_cycles_write_result<<std::endl;
            // std::cout<<"write all result need cycles : "<<n_cycles_write_result<<std::endl;
            std::cout<<"The output has not accumulated enough and will not be entered into DRAM"<<std::endl;
        }
        
        if((i == num_weight_buffer_fold-1) && (num_output_write_once != 0)){
            n_cycles_write_result = store_output_and_neuronstate_data_fc(ofmap, nfmap, dram_instance, num_output_obtained, num_output_write_once, layer_parameters);
            num_output_obtained += num_output_write_once;
            // std::cout<<"n_cycles_write_result : "<<n_cycles_write_result<<std::endl;
            std::cout<<"write all result need cycles : "<<n_cycles_write_result<<std::endl;
            this->n_cycles += n_cycles_write_result;
            std::cout<<"All calculation ends, and write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
            num_output_write_once = 0;
        } 

    }

    // 计算完毕，验证写到DRAM的结果是否正确
    matrixMultiply(output_layer, input_layer, 1, filter, ifmap, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th, this->pooling_enabled);

    for(int i=0; i<output_layer; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;

    delete dram_instance;
    delete[] this->output_buffer;
    // delete[] this->output_buffer_cpu;
    delete[] this->neuron_state_buffer;
    // delete[] this->neuron_state_buffer_cpu;
    delete this->ppbuf_input;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;
    delete this->ppbuf_weight;
    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    // delete[] acc_inter_output;
    // delete[] acc_inter_result;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}


std::tuple<int*, int*, int*, int*> Controller::runConv_CHW(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                     Call runConv_CHW function                                             "<<std::endl;
    std::cout<<"                                     Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    //std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m       "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            // std::cout<<"i : "<<i<<std::endl;
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
            // std::cout<<r.num_rows<<std::endl;
            // std::cout<<r.start_rows<<std::endl;
            // std::cout<<r.add_0_above<<std::endl;
            // std::cout<<r.add_0_below<<std::endl;
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    // 1. weight buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    // 从DRAM中取数据需要的（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用
    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于计算下次取数据的基地址
    int num_weight_obtained = 0;
    // 第一次从DRAM中加载权重数据，加载到next_buffer中
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    this->n_cycles += n_cycles_load_first_weight;
    num_weight_obtained += num_weight_data;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);  // 交换缓冲区
    std::cout<<"load the first weights need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 2. input buffer
    int num_input_buffer_need = R * Y_padded * C;  // 加padding
    assert(num_input_buffer_need <= this->num_input);
    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    // 第一次加载input data
    // 加载数据到用于计算的buffer，加载到next_buffer
    int n_cycles_load_first_input = load_input_data_CHW(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
    // 切换，切换之后，上一步加载的数据到了current_buffer，用于下面的计算，next_buffer是空的，用于加载下一块数据（和计算同时进行）
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
    this->n_cycles += n_cycles_load_first_input;
    std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 3. output buffer
    int num_output_buffer_need = X_*Y_ * std::min<int>(this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);  // 卷积的输出
    assert(num_output_buffer_need <= this->num_output);
    // this->ppbuf_output = new PingPong_Buffer;
    // this->output_buffer_0 = new int[num_output_buffer_need];
    // this->output_buffer_1 = new int[num_output_buffer_need];
    // PingPongBuffer_Init(this->ppbuf_output, this->output_buffer_0, this->output_buffer_1);
    this->output_buffer = new int[num_output_buffer_need];

    // 4. neuron_state buffer
    int num_neuron_state_buffer_need = X_*Y_ * std::min<int>(this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);  
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);
    // this->ppbuf_neuron_state = new PingPong_Buffer;
    // this->neuron_state_buffer_0 = new int[num_neuron_state_buffer_need];
    // this->neuron_state_buffer_1 = new int[num_neuron_state_buffer_need];
    // PingPongBuffer_Init(this->ppbuf_neuron_state, this->neuron_state_buffer_0, this->neuron_state_buffer_1);
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need];

    // 下面开始进入计算循环
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;
        
        for(int j=0; j<X_; j++){ 
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            
            // 加载下一块数据到next_buffer中
            // 如果（j+1==X_）加载的下一块数据是下一个weight循环的第一块数据，但要保证有下一块权重的计算
            int n_cycles_load_next_input;
            if((j+1)<X_ || (i+1)<num_weight_buffer_fold) {
                n_cycles_load_next_input = load_input_data_CHW(layer_id, ifmap, this->dram_instance, (j+1)%X_, layer_parameters);  
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // 在第二次从DRAM中取输入数据之后，与计算同时进行加载下一块权重数据
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){  // 取下一块权重
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
                num_weight_obtained += num_weight_data;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }
            
            // *** input数据重排和调用脉动阵列进行计算，在该函数内，将计算的结果累积到output buffer中
            int n_cycles_process = process_conv_CHW(layer_id, i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_process;
            std::cout<<"process_conv need cycles : "<<n_cycles_process<<std::endl;

            this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process);
            std::cout<<"The current input loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
            //std::cout<<std::endl;

            // input双缓冲区切换
            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
            
        }

        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); // 切换缓冲区

        // 将当前权重循环得到的输出结果写入DRAM
        int n_cycles_write_neuron_state = store_neuron_state_CHW(nfmap, dram_instance, i, cols, layer_parameters);
        int n_cycles_write_output = store_conv_output_CHW(ofmap, dram_instance, i, cols, layer_parameters);
        std::cout<<std::endl;
        std::cout<<"write neuron state result need cycles : "<<n_cycles_write_neuron_state<<std::endl;
        std::cout<<"write output result need cycles : "<<n_cycles_write_output<<std::endl;
        std::cout<<"write all result need cycles : "<<(n_cycles_write_output + n_cycles_write_neuron_state)<<std::endl;

        if(i == num_weight_buffer_fold-1){
            this->n_cycles += n_cycles_write_output;
            this->n_cycles += n_cycles_write_neuron_state;
            std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
        } else {
            std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
        }
    }

    delete this->ppbuf_weight;
    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete this->ppbuf_input;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;

    delete[] this->output_buffer;
    delete[] this->neuron_state_buffer;

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    //std::cout<<"begin final test"<<std::endl;
    conv_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    for(int i=0; i<X_*Y_*K; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    // std::cout<<std::endl;
    // std::cout<<"The conv compute output result is : "<<std::endl;
    // for(int k=0; k<K; k++){
    //     std::cout<<"output channel : "<<std::endl;
    //     for(int p=0; p<X_; p++){
    //         for(int q=0; q<Y_; q++){
    //             std::cout<<ofmap[k*X_*Y_ + p*Y_ +q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // std::cout<<std::endl;
    // std::cout<<"The conv compute neuron_state result is : "<<std::endl;
    // for(int k=0; k<K; k++){
    //     std::cout<<"output channel : "<<std::endl;
    //     for(int p=0; p<X_; p++){
    //         for(int q=0; q<Y_; q++){
    //             std::cout<<nfmap[k*X_*Y_ + p*Y_ +q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    this->records.clear();

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

std::tuple<int*, int*, int*, int*> Controller::runConv_HWC(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    // 通道后置
    // 从DRAM中读取输入数据是连续的
    // 将输出数据写入DRAM时，非常不连续
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                     Call runConv_HWC function                                             "<<std::endl;
    std::cout<<"                                     Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    // std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m       "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            // std::cout<<"i : "<<i<<std::endl;
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
            // std::cout<<r.num_rows<<std::endl;
            // std::cout<<r.start_rows<<std::endl;
            // std::cout<<r.add_0_above<<std::endl;
            // std::cout<<r.add_0_below<<std::endl;
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // 1. weight buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    // 从DRAM中取数据需要的（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用
    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于计算下次取数据的基地址
    int num_weight_obtained = 0;
    // 第一次从DRAM中加载权重数据，加载到next_buffer中
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    this->n_cycles += n_cycles_load_first_weight;
    num_weight_obtained += num_weight_data;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);  // 交换缓冲区
    std::cout<<"load the first weights need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 2. input buffer
    int num_input_buffer_need = R * Y_padded * C;  // 加padding
    assert(num_input_buffer_need <= this->num_input);
    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    // 第一次加载input data
    // 加载数据到用于计算的buffer，加载到next_buffer
    int n_cycles_load_first_input = load_input_data_HWC(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
    // 切换，切换之后，上一步加载的数据到了current_buffer，用于下面的计算，next_buffer是空的，用于加载下一块数据（和计算同时进行）
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
    this->n_cycles += n_cycles_load_first_input;
    std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 3. output buffer
    // 注意output buffer的大小，这和通道后置存储方式相关
    int num_output_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols;  // 卷积的输出
    assert(num_output_buffer_need <= this->num_output);
    this->output_buffer = new int[num_output_buffer_need]();
    this->output_buffer_cpu = new int[num_output_buffer_need]();

    // 4. neuron_state buffer
    int num_neuron_state_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    for(int i=0; i<num_weight_buffer_fold; i++){
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;  

        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            
            // 加载下一块数据到next_buffer中
            // 如果（j+1==X_）加载的下一块数据是下一个weight循环的第一块数据，但要保证有下一块权重的计算
            int n_cycles_load_next_input;
            if((j+1)<X_ || (i+1)<num_weight_buffer_fold) {
                n_cycles_load_next_input = load_input_data_HWC(layer_id, ifmap, this->dram_instance, (j+1)%X_, layer_parameters);  
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // 在第二次从DRAM中取输入数据之后，与计算同时进行加载下一块权重数据
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){  // 取下一块权重
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
                num_weight_obtained += num_weight_data;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // input数据重排和调用脉动阵列进行计算，在该函数内，将计算的结果累积到output buffer中
            int n_cycles_process = process_conv_HWC(layer_id, i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_process;
            std::cout<<"process_conv need cycles : "<<n_cycles_process<<std::endl;

            this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process);
            std::cout<<"The current input loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
            std::cout<<std::endl;

            // 将计算结果写入DRAM
            int n_cycles_write_output = store_conv_output_HWC(ofmap, this->dram_instance, i, j, cols, layer_parameters);
            int n_cycles_write_neuron_state = store_neuron_state_HWC(nfmap, dram_instance, i, j, cols, layer_parameters);
            std::cout<<"write neuron state result need cycles : "<<n_cycles_write_neuron_state<<std::endl;
            std::cout<<"write output result need cycles : "<<n_cycles_write_output<<std::endl;
            std::cout<<"write all result need cycles : "<<(n_cycles_write_output + n_cycles_write_neuron_state)<<std::endl;

            // 加上最后一次写数据所需的周期
            if(i==num_weight_buffer_fold-1 && j==X_-1){
                this->n_cycles += n_cycles_write_output;
                this->n_cycles += n_cycles_write_neuron_state;
                std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
            } else {
                std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
            }

            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        }
        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); // 切换缓冲区

    }

    delete this->ppbuf_weight;
    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete this->ppbuf_input;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;

    delete[] this->output_buffer;
    delete[] this->output_buffer_cpu;
    delete[] this->neuron_state_buffer;
    delete[] this->neuron_state_buffer_cpu;

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    //std::cout<<"begin final test"<<std::endl;
    conv_compute_HWC(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    for(int i=0; i<X_*Y_*K; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;

    this->records.clear();
    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

std::tuple<int*, int*, int*, int*> Controller::runConv_HCW(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    // HCW
    // 通道后置和通道前置的折中
    // 从DRAM中读取输入数据是连续的
    // 将输出数据写入DRAM时，不完全连续，但是相比通道前置可以减少写入次数

    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                     Call runConv_HCW function                                             "<<std::endl;
    std::cout<<"                                     Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    // std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m       "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            // std::cout<<"i : "<<i<<std::endl;
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
            // std::cout<<r.num_rows<<std::endl;
            // std::cout<<r.start_rows<<std::endl;
            // std::cout<<r.add_0_above<<std::endl;
            // std::cout<<r.add_0_below<<std::endl;
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    // std::cout<<"below is ifmap data : "<<std::endl;
    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<X; p++){
    //         for(int q=0; q<Y; q++){
    //             // int index = p * Y * C + q * C + channels;  // HWC layout
    //             int index = p * C * Y + channels * Y + q; // HCW layout
    //             std::cout<<ifmap[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // 1. weight buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    // 从DRAM中取数据需要的（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用
    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于计算下次取数据的基地址
    int num_weight_obtained = 0;
    // 第一次从DRAM中加载权重数据，加载到next_buffer中
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    this->n_cycles += n_cycles_load_first_weight;
    num_weight_obtained += num_weight_data;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);  // 交换缓冲区
    std::cout<<"load the first weights need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 2. input buffer
    int num_input_buffer_need = R * Y_padded * C;  // 加padding
    assert(num_input_buffer_need <= this->num_input);
    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    // 第一次加载input data
    // 加载数据到用于计算的buffer，加载到next_buffer
    int n_cycles_load_first_input = load_input_data_HCW(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
    // 切换，切换之后，上一步加载的数据到了current_buffer，用于下面的计算，next_buffer是空的，用于加载下一块数据（和计算同时进行）
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
    this->n_cycles += n_cycles_load_first_input;
    std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 3. output buffer
    int num_output_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols;  // 卷积的输出
    assert(num_output_buffer_need <= this->num_output);
    this->output_buffer = new int[num_output_buffer_need]();
    // this->output_buffer_cpu = new int[num_output_buffer_need]();

    // 4. neuron_state buffer
    int num_neuron_state_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    // this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    for(int i=0; i<num_weight_buffer_fold; i++){
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;  

        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            
            // 加载下一块数据到next_buffer中
            // 如果（j+1==X_）加载的下一块数据是下一个weight循环的第一块数据，但要保证有下一块权重的计算
            int n_cycles_load_next_input;
            if((j+1)<X_ || (i+1)<num_weight_buffer_fold) {
                n_cycles_load_next_input = load_input_data_HCW(layer_id, ifmap, this->dram_instance, (j+1)%X_, layer_parameters);  
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // 在第二次从DRAM中取输入数据之后，与计算同时进行加载下一块权重数据
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){  // 取下一块权重
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
                num_weight_obtained += num_weight_data;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // 计算
            int n_cycles_process = process_conv_HCW(layer_id, i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_process;
            std::cout<<"process_conv need cycles : "<<n_cycles_process<<std::endl;

            this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process);
            std::cout<<"The current input loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
            std::cout<<std::endl;

            // 将计算结果写入DRAM
            int n_cycles_write_output = store_conv_output_HCW(ofmap, this->dram_instance, i, j, cols, layer_parameters);
            int n_cycles_write_neuron_state = store_neuron_state_HCW(nfmap, dram_instance, i, j, cols, layer_parameters);
            std::cout<<"write neuron state result need cycles : "<<n_cycles_write_neuron_state<<std::endl;
            std::cout<<"write output result need cycles : "<<n_cycles_write_output<<std::endl;
            std::cout<<"write all result need cycles : "<<(n_cycles_write_output + n_cycles_write_neuron_state)<<std::endl;

            // 加上最后一次写数据所需的周期
            if(i==num_weight_buffer_fold-1 && j==X_-1){
                this->n_cycles += n_cycles_write_output;
                this->n_cycles += n_cycles_write_neuron_state;
                std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
            } else {
                std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
            }

            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        }
        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); // 切换缓冲区
    }

    delete this->ppbuf_weight;
    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete this->ppbuf_input;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;

    delete[] this->output_buffer;
    // delete[] this->output_buffer_cpu;
    delete[] this->neuron_state_buffer;
    // delete[] this->neuron_state_buffer_cpu;

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    //std::cout<<"begin final test"<<std::endl;
    conv_compute_HCW(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    for(int i=0; i<X_*Y_*K; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;

    this->records.clear();
    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

std::tuple<int*, int*, int*, int*> Controller::runConv_HCW_bank(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                     Call runConv_HCW_bank function                                             "<<std::endl;
    std::cout<<"                                     Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    assert(R == 3); // 目前只支持卷积核大小为3*3的卷积层

    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // for(int i=0; i<X_; i++){
    //     record r;
    //     if(i>=0 && i<P){
    //         r.start_rows = 0;
    //         r.num_rows = R-(P-i);
    //         r.add_0_above = P-i;
    //         r.add_0_below = 0;
    //     } else if(i>X_-1-P && i<=X_-1){
    //         // std::cout<<"i : "<<i<<std::endl;
    //         r.start_rows = i-P;
    //         r.num_rows = R - (i-(X_-1-P));
    //         r.add_0_above = 0;
    //         r.add_0_below = (i-(X_-1-P));
    //         // std::cout<<r.num_rows<<std::endl;
    //         // std::cout<<r.start_rows<<std::endl;
    //         // std::cout<<r.add_0_above<<std::endl;
    //         // std::cout<<r.add_0_below<<std::endl;
    //     } else {
    //         r.start_rows = i-P;
    //         r.num_rows = R;
    //         r.add_0_above = 0;
    //         r.add_0_below = 0;
    //     }
    //     this->records.push_back(r);
    // }


    // ------------------------------------------------------------------------------------------------------------------------------------------------------------

    // std::cout<<"below is ifmap data : "<<std::endl;
    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<X; p++){
    //         for(int q=0; q<Y; q++){
    //             // int index = p * Y * C + q * C + channels;  // HWC layout
    //             int index = p * C * Y + channels * Y + q; // HCW layout
    //             std::cout<<ifmap[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // 1. weight buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    // 从DRAM中取数据需要的（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用
    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于计算下次取数据的基地址
    int num_weight_obtained = 0;
    // 第一次从DRAM中加载权重数据，加载到next_buffer中
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    this->n_cycles += n_cycles_load_first_weight;
    std::cout<<"load the first weights need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;
    num_weight_obtained += num_weight_data;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);  // 交换缓冲区

    // 2. input buffer 
    int num_input_bank_need = Y_padded*C;
    assert(num_input_bank_need*(R+1) <= this->num_input);
    this->input_base_bank_0 = new int[num_input_bank_need];
    this->input_base_bank_1 = new int[num_input_bank_need];
    this->ppbuf_bank = new PingPong_Buffer;
    this->input_pp_bank_0 = new int[num_input_bank_need];
    this->input_pp_bank_1 = new int[num_input_bank_need];
    PingPongBuffer_Init(this->ppbuf_bank, this->input_pp_bank_0, this->input_pp_bank_1);

    // 3. output buffer
    int num_output_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols;  // 卷积的输出
    assert(num_output_buffer_need <= this->num_output);
    this->output_buffer = new int[num_output_buffer_need]();
    // this->output_buffer_cpu = new int[num_output_buffer_need]();

    // 4. neuron_state buffer
    int num_neuron_state_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    // this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    for(int i=0; i<num_weight_buffer_fold; i++){
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;  

        // 第一次加载input data
        int n_cycles_load_first_input = load_input_data_HCW_bank(layer_id, ifmap, dram_instance, 0, layer_parameters);
        PingPongBuffer_Switch(this->ppbuf_bank, this->input_pp_bank_0, this->input_pp_bank_1);
        this->n_cycles += n_cycles_load_first_input;
        std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            
            // *** 加载下一块输入数据，与上一块数据的计算并行
            int n_cycles_load_next_input;
            if((j+1)<X_ ) {
                n_cycles_load_next_input = load_input_data_HCW_bank(layer_id, ifmap, this->dram_instance, j+1, layer_parameters);  
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // *** 加载下一块权重数据，与当前权重循环的计算过程并行
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){  // 取下一块权重
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
                num_weight_obtained += num_weight_data;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            /// *** 重排和计算
            int n_cycles_process = process_conv_HCW_bank(layer_id, i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_process;
            std::cout<<"process_conv need cycles : "<<n_cycles_process<<std::endl;

            this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process);
            std::cout<<"The current input loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
            std::cout<<std::endl;

            // *** 将计算结果写入DRAM
            int n_cycles_write_output = store_conv_output_HCW(ofmap, this->dram_instance, i, j, cols, layer_parameters);
            int n_cycles_write_neuron_state = store_neuron_state_HCW(nfmap, dram_instance, i, j, cols, layer_parameters);
            std::cout<<"write neuron state result need cycles : "<<n_cycles_write_neuron_state<<std::endl;
            std::cout<<"write output result need cycles : "<<n_cycles_write_output<<std::endl;
            std::cout<<"write all result need cycles : "<<(n_cycles_write_output + n_cycles_write_neuron_state)<<std::endl;

            // 加上最后一次写数据所需的周期
            if(i==num_weight_buffer_fold-1 && j==X_-1){
                // std::cout<<"---- The last time write data into DRAM ------"<<std::endl;
                this->n_cycles += n_cycles_write_output;
                this->n_cycles += n_cycles_write_neuron_state;
                std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
            } else {
                std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
            }

            PingPongBuffer_Switch(this->ppbuf_bank, this->input_pp_bank_0, this->input_pp_bank_1);
        }
        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    }

    delete this->ppbuf_weight;
    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;

    delete this->ppbuf_bank;
    delete[] this->input_base_bank_0;
    delete[] this->input_base_bank_1;
    delete[] this->input_pp_bank_0;
    delete[] this->input_pp_bank_1;

    delete[] this->output_buffer;
    delete[] this->neuron_state_buffer;

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    //std::cout<<"begin final test"<<std::endl;
    conv_compute_HCW(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    for(int i=0; i<X_*Y_*K; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }
    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;

    // this->records.clear();
    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}


std::tuple<int*, int*, int*, int*> Controller::runConvandPooling_CHW(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积

    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                  Call runConvandPooling_CHW function                                    "<<std::endl;
    std::cout<<"                                  Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    //std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m         "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长
    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K / 4;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 加padding操作在从DRAM中取数据到input_buffer这个阶段完成，在load_input中所需要的控制信号
    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------------

    // 从DRAM中取数据需要的一些（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 1. weight buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    int num_weight_obtained = 0;  // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    // 第一次从DRAM中取权重数据
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    num_weight_obtained += num_weight_data;
    this->n_cycles += n_cycles_load_first_weight;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);
    std::cout<<"load the first weights need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;
    
    // 2. input buffer
    int num_input_buffer_need = R * Y_padded * C;
    assert(num_input_buffer_need <= this->num_input);
    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_arranged_buffer_1);

    // 第一次加载输入数据到片上buffer
    int n_cycles_load_first_input = load_input_data_CHW(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
    this->n_cycles += n_cycles_load_first_input;
    std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    // 3. output buffer
    int num_output_buffer_need = X_*Y_/4 * std::min<int>(this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);  // 卷积的输出，后接池化模块
    assert(num_output_buffer_need <= this->num_output);
    this->output_buffer = new int[num_output_buffer_need];

    // 4. neuron state buffer
    int num_neuron_state_buffer_need = X_*Y_ * std::min<int>(this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);  
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need];

    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 建模片上SRAM，用于累积卷积的输出进行池化
        int sram_size = 2 * Y_ * cols; 
        this->on_chip_sram = new int[sram_size]();

        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;

            // *** 加载下一块输入数据，与上一块数据的计算并行
            int n_cycles_load_next_input;
            if((j+1)<X_ || (i+1)<num_weight_buffer_fold) {
                n_cycles_load_next_input = load_input_data_CHW(layer_id, ifmap, this->dram_instance, (j+1)%X_, layer_parameters);
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // *** 加载下一块权重数据，与当前权重循环的计算过程并行
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                num_weight_obtained += num_weight_data;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // *** 重排和计算，将neuron state写入片上buffer，将output累积到片上sram
            int n_cycles_process = process_conv_and_pooling_CHW(layer_id, i, j, cols, count_rows, layer_parameters);
            count_rows++;
            std::cout<<"process_conv_and_pooling need cycles : "<<n_cycles_process<<std::endl;

            //对累积的结果进行池化，将池化结果写入片上buffer
            int n_cycles_process_pooling;
            if(j>0 && j%2!=0){
                count_rows = 0;
                // 调用池化模块
                n_cycles_process_pooling = process_pooling_CHW(i, j, cols, layer_parameters);
                std::cout<<"process_pooling need cycles : "<<n_cycles_process_pooling<<std::endl;

            } else {
                n_cycles_process_pooling = 0;
            }

            this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_process_pooling);
            std::cout<<"The current input loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;

            // 切换输入缓冲区
            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
            //std::cout<<"debug"<<std::endl;
        }

        // 切换权重缓冲区
        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); 

        // 将当前权重循环得到的输出结果写入DRAM
        int n_cycles_write_neuron_state = store_neuron_state_CHW(nfmap, dram_instance, i, cols, layer_parameters);
        int n_cycles_write_output = store_pooling_output_CHW(ofmap, dram_instance, i, cols, layer_parameters);
        std::cout<<std::endl;
        std::cout<<"write neuron state result cycles : "<<n_cycles_write_neuron_state<<std::endl;
        std::cout<<"write output result cycles : "<<n_cycles_write_output<<std::endl;
        std::cout<<"write all result need cycles : "<<(n_cycles_write_output + n_cycles_write_neuron_state)<<std::endl;

        if(i == num_weight_buffer_fold-1){
            this->n_cycles += n_cycles_write_neuron_state;
            this->n_cycles += n_cycles_write_output;
            std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
        } else {
            std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
        }

        delete[] this->on_chip_sram;
    }

    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete this->ppbuf_weight;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;
    delete[] this->ppbuf_input;

    delete[] this->output_buffer;
    delete[] this->neuron_state_buffer;

    // std::cout<<std::endl;
    // std::cout<<"The conv compute output result is : "<<std::endl;
    // for(int k=0; k<K; k++){
    //     std::cout<<"output channel : "<<k<<std::endl;
    //     for(int p=0; p<X_/2; p++){
    //         for(int q=0; q<Y_/2; q++){
    //             std::cout<<ofmap[k*X_*Y_/4 + p*Y_/2 +q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_and_pooling_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查神经元状态
    for(int i=0; i<X_*Y_*K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

        // 检查输出
    for(int i=0; i<X_*Y_*K/4; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;

    this->records.clear();

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

std::tuple<int*, int*, int*, int*> Controller::runConvandPooling_HWC(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                  Call runConvandPooling_HWC function                                    "<<std::endl;
    std::cout<<"                                  Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;
    
    // std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m         "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

     // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长
    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K / 4;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 加padding操作在从DRAM中取数据到input_buffer这个阶段完成，在load_input中所需要的控制信号
    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------------

    // std::cout<<"below is ifmap data : "<<std::endl;
    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<X; p++){
    //         for(int q=0; q<Y; q++){
    //             int index = p * Y * C + q * C + channels;  // HWC layout
    //             // int index = p * C * Y + channels * Y + q; // HCW layout
    //             std::cout<<ifmap[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }


    // 从DRAM中取数据需要的一些（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 1. weight buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    int num_weight_obtained = 0;  // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    // 第一次从DRAM中取权重数据
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    num_weight_obtained += num_weight_data;
    this->n_cycles += n_cycles_load_first_weight;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);
    std::cout<<"load the first weights need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;
    
    // 2. input buffer
    int num_input_buffer_need = R * Y_padded * C;
    assert(num_input_buffer_need <= this->num_input);
    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_arranged_buffer_1);

    // 第一次加载输入数据到片上buffer
    int n_cycles_load_first_input = load_input_data_HWC(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
    this->n_cycles += n_cycles_load_first_input;
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
    std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 3. output buffer
    int num_output_buffer_need = Y_/2 * this->stonne_cfg.m_MSNetworkCfg.ms_cols;  // 卷积的输出，后接池化模块
    assert(num_output_buffer_need <= this->num_output);
    this->output_buffer = new int[num_output_buffer_need]();

    // 4. neuron state buffer
    int num_neuron_state_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    for(int i=0; i<num_weight_buffer_fold; i++){
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;  

        // 建模片上SRAM，用于累积卷积的输出进行池化
        int sram_size = 2 * Y_ * cols; 
        this->on_chip_sram = new int[sram_size]();

        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;

            // *** 加载下一块输入数据，与上一块数据的计算并行
            int n_cycles_load_next_input;
            if((j+1)<X_ || (i+1)<num_weight_buffer_fold) {
                n_cycles_load_next_input = load_input_data_HWC(layer_id, ifmap, this->dram_instance, (j+1)%X_, layer_parameters);
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // *** 加载下一块权重数据，与当前权重循环的计算过程并行
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                num_weight_obtained += num_weight_data;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // *** 重排和计算，将neuron state写入片上buffer，将output累积到片上sram
            int n_cycles_process = process_conv_and_pooling_HWC(layer_id, i, j, cols, count_rows, layer_parameters);
            count_rows++;
            std::cout<<"process_conv_and_pooling need cycles : "<<n_cycles_process<<std::endl;

            // 1. 将神经元状态写入DRAM
            int n_cycles_write_neuron_state = store_neuron_state_HWC(nfmap, this->dram_instance, i, j, cols, layer_parameters);
            // std::cout<<"write neuron_state cycles : "<<n_cycles_write_neuron_state<<std::endl;

            // 2. 对累计的输出进行池化
            int n_cycles_process_pooling;
            int n_cycles_write_output;
            if(j>0 && j%2!=0){
                count_rows=0;

                // 调用池化模块
                n_cycles_process_pooling = process_pooling_HWC(i, j, cols, layer_parameters);
                std::cout<<"process_pooling need cycles : "<<n_cycles_process_pooling<<std::endl;

                // 将池化结果写入DRAM
                n_cycles_write_output = store_pooling_output_HWC(ofmap, dram_instance, i, j, cols, layer_parameters);
                //std::cout<<"write output cycles : "<<n_cycles_write_output<<std::endl;
            } else {
                n_cycles_process_pooling = 0;
                n_cycles_write_output = 0;
            }

            this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_process_pooling);
            std::cout<<"The current input loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
            std::cout<<std::endl;

            std::cout<<"write neuron_state need cycles : "<<n_cycles_write_neuron_state<<std::endl;
            std::cout<<"write output need cycles : "<<n_cycles_write_output<<std::endl;
            std::cout<<"write all result need cycles : "<<(n_cycles_write_output + n_cycles_write_neuron_state)<<std::endl;

            // 加上最后一次写数据所需的周期
            if(i==num_weight_buffer_fold-1 && j==X_-1){
                this->n_cycles += n_cycles_write_output;
                this->n_cycles += n_cycles_write_neuron_state;
                std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
            } else {
                std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
            }

            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        }

        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); 
        delete[] this->on_chip_sram;
    }

    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete this->ppbuf_weight;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;
    delete[] this->ppbuf_input;

    delete[] this->output_buffer;
    delete[] this->neuron_state_buffer;
    delete[] this->neuron_state_buffer_cpu;

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_and_pooling_compute_HWC(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查神经元状态
    for(int i=0; i<X_*Y_*K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

        // 检查输出
    for(int i=0; i<X_*Y_*K/4; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;

    // std::cout<<"below is ofmap data : "<<std::endl;
    // for(int channels=0; channels<K; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<X_/2; p++){
    //         for(int q=0; q<Y_/2; q++){
    //             int index = p * Y_/2 * K + q * K + channels;  // HWC layout
    //             // int index = p * C * Y + channels * Y + q; // HCW layout
    //             std::cout<<ofmap[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    this->records.clear();

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

std::tuple<int*, int*, int*, int*> Controller::runConvandPooling_HCW(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                  Call runConvandPooling_HCW function                                    "<<std::endl;
    std::cout<<"                                  Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;
    
    // std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m         "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长
    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K / 4;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 加padding操作在从DRAM中取数据到input_buffer这个阶段完成，在load_input中所需要的控制信号
    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------------
    // std::cout<<"below is ifmap data : "<<std::endl;
    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<X; p++){
    //         for(int q=0; q<Y; q++){
    //             int index = p * Y * C + q * C + channels;  // HWC layout
    //             // int index = p * C * Y + channels * Y + q; // HCW layout
    //             std::cout<<ifmap[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // 从DRAM中取数据需要的一些（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 1. weight buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    int num_weight_obtained = 0;  // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    // 第一次从DRAM中取权重数据
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    num_weight_obtained += num_weight_data;
    this->n_cycles += n_cycles_load_first_weight;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);
    std::cout<<"load the first weights need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;
    
    // 2. input buffer
    int num_input_buffer_need = R * Y_padded * C;
    assert(num_input_buffer_need <= this->num_input);
    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_arranged_buffer_1);

    // 第一次加载输入数据到片上buffer
    int n_cycles_load_first_input = load_input_data_HCW(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
    this->n_cycles += n_cycles_load_first_input;
    std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    // 3. output buffer
    // 池化后的结果不需要再进行转置，可以直接写入DRAM
    int num_output_buffer_need = Y_/2 * this->stonne_cfg.m_MSNetworkCfg.ms_cols;  // 卷积的输出，后接池化模块
    assert(num_output_buffer_need <= this->num_output);
    this->output_buffer = new int[num_output_buffer_need]();
    this->output_buffer_cpu = new int[num_output_buffer_need]();

    // 4. neuron state buffer
    // 计算结果需要转置
    int num_neuron_state_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    // this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    for(int i=0; i<num_weight_buffer_fold; i++){
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;  

        // 建模片上SRAM，用于累积卷积的输出进行池化
        int sram_size = 2 * Y_ * cols; 
        this->on_chip_sram = new int[sram_size]();

        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;

            // *** 加载下一块输入数据，与上一块数据的计算并行
            int n_cycles_load_next_input;
            if((j+1)<X_ || (i+1)<num_weight_buffer_fold) {
                n_cycles_load_next_input = load_input_data_HCW(layer_id, ifmap, this->dram_instance, (j+1)%X_, layer_parameters);
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // *** 加载下一块权重数据，与当前权重循环的计算过程并行
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                num_weight_obtained += num_weight_data;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // *** 重排和计算，将neuron state写入片上buffer，将output累积到片上sram
            int n_cycles_process = process_conv_and_pooling_HCW(layer_id, i, j, cols, count_rows, layer_parameters);
            count_rows++;
            std::cout<<"process_conv_and_pooling need cycles : "<<n_cycles_process<<std::endl;

            // 1. 将神经元状态写入DRAM
            int n_cycles_write_neuron_state = store_neuron_state_HCW(nfmap, this->dram_instance, i, j, cols, layer_parameters);
            // std::cout<<"write neuron_state cycles : "<<n_cycles_write_neuron_state<<std::endl;

            // 2. 对累计的输出进行池化
            int n_cycles_process_pooling;
            int n_cycles_write_output;
            if(j>0 && j%2!=0){
                count_rows=0;

                // 调用池化模块
                n_cycles_process_pooling = process_pooling_HCW(i, j, cols, layer_parameters);
                std::cout<<"process_pooling need cycles : "<<n_cycles_process_pooling<<std::endl;

                // 将池化结果写入DRAM
                n_cycles_write_output = store_pooling_output_HCW(ofmap, dram_instance, i, j, cols, layer_parameters);
                // std::cout<<"write output cycles : "<<n_cycles_write_output<<std::endl;
            } else {
                n_cycles_process_pooling = 0;
                n_cycles_write_output = 0;
            }

            this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_process_pooling);
            std::cout<<"The current input loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
            std::cout<<std::endl;

            std::cout<<"write neuron_state need cycles : "<<n_cycles_write_neuron_state<<std::endl;
            std::cout<<"write output need cycles : "<<n_cycles_write_output<<std::endl;
            std::cout<<"write all result need cycles : "<<(n_cycles_write_output + n_cycles_write_neuron_state)<<std::endl;

            // 加上最后一次写数据所需的周期
            if(i==num_weight_buffer_fold-1 && j==X_-1){
                this->n_cycles += n_cycles_write_output;
                this->n_cycles += n_cycles_write_neuron_state;
                std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
            } else {
                std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
            }

            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        }

        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); 
        delete[] this->on_chip_sram;
    }

    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete this->ppbuf_weight;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;
    delete[] this->ppbuf_input;

    delete[] this->output_buffer;
    delete[] this->output_buffer_cpu;
    delete[] this->neuron_state_buffer;
    // delete[] this->neuron_state_buffer_cpu;

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_and_pooling_compute_HCW(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查神经元状态
    for(int i=0; i<X_*Y_*K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    // 检查输出
    for(int i=0; i<X_*Y_*K/4; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;

    this->records.clear();

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

std::tuple<int*, int*, int*, int*> Controller::runConvandPooling_HCW_bank(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                  Call runConvandPooling_HCW_bank function                                    "<<std::endl;
    std::cout<<"                                  Start simulation layer : "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;
    
    // std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m         "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长
    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K / 4;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // std::cout<<"below is ifmap data : "<<std::endl;
    // for(int channels=0; channels<C; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<X; p++){
    //         for(int q=0; q<Y; q++){
    //             // int index = p * Y * C + q * C + channels;  // HWC layout
    //             int index = p * C * Y + channels * Y + q; // HCW layout
    //             std::cout<<ifmap[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // -------------------------------------------------------------------------------------------------------------------------------------------------

    // 从DRAM中取数据需要的一些（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 1. weight buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    int num_weight_obtained = 0;  // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    // 第一次从DRAM中取权重数据
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    num_weight_obtained += num_weight_data;
    this->n_cycles += n_cycles_load_first_weight;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);
    std::cout<<"load the first weights need cycles : "<<n_cycles_load_first_weight<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

    // 2. input buffer
    int num_input_bank_need = Y_padded*C;
    assert(num_input_bank_need*(R+1) <= this->num_input);
    this->input_base_bank_0 = new int[num_input_bank_need];
    this->input_base_bank_1 = new int[num_input_bank_need];
    this->ppbuf_bank = new PingPong_Buffer;
    this->input_pp_bank_0 = new int[num_input_bank_need];
    this->input_pp_bank_1 = new int[num_input_bank_need];
    PingPongBuffer_Init(this->ppbuf_bank, this->input_pp_bank_0, this->input_pp_bank_1);

    // 3. output buffer
    // 池化后的结果不需要再进行转置，可以直接写入DRAM
    int num_output_buffer_need = Y_/2 * this->stonne_cfg.m_MSNetworkCfg.ms_cols;  // 卷积的输出，后接池化模块
    assert(num_output_buffer_need <= this->num_output);
    this->output_buffer = new int[num_output_buffer_need]();
    this->output_buffer_cpu = new int[num_output_buffer_need]();

    // 4. neuron state buffer
    // 计算结果需要转置
    int num_neuron_state_buffer_need = Y_ * this->stonne_cfg.m_MSNetworkCfg.ms_cols; 
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    // this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    for(int i=0; i<num_weight_buffer_fold; i++){
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;  

        // 建模片上SRAM，用于累积卷积的输出进行池化
        int sram_size = 2 * Y_ * cols; 
        this->on_chip_sram = new int[sram_size]();

        // 第一次加载input data
        int n_cycles_load_first_input = load_input_data_HCW_bank(layer_id, ifmap, dram_instance, 0, layer_parameters);
        PingPongBuffer_Switch(this->ppbuf_bank, this->input_pp_bank_0, this->input_pp_bank_1);
        this->n_cycles += n_cycles_load_first_input;
        std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<"     and the current global cycles is : "<<this->n_cycles<<std::endl;

        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            //std::cout<<"current input loop begin cycles : "<<this->n_cycles<<"          and current cycles is : "<<this->n_cycles<<std::endl;

            // *** 加载下一块输入数据，与上一块数据的计算并行
            int n_cycles_load_next_input;
            if((j+1)<X_) {
                n_cycles_load_next_input = load_input_data_HCW_bank(layer_id, ifmap, this->dram_instance, j+1, layer_parameters);
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input need cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // *** 加载下一块权重数据，与当前权重循环的计算过程并行
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                num_weight_obtained += num_weight_data;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // *** 重排和计算，将neuron state写入片上buffer，将output累积到片上sram
            int n_cycles_process = process_conv_and_pooling_HCW_bank(layer_id, i, j, cols, count_rows, layer_parameters);
            count_rows++;
            std::cout<<"process_conv_and_pooling need cycles : "<<n_cycles_process<<std::endl;

            // 1. 将神经元状态写入DRAM
            int n_cycles_write_neuron_state = store_neuron_state_HCW(nfmap, this->dram_instance, i, j, cols, layer_parameters);
            //std::cout<<"write neuron_state need cycles : "<<n_cycles_write_neuron_state<<std::endl;

            // 2. 对累计的输出进行池化
            int n_cycles_process_pooling;
            int n_cycles_write_output;
            if(j>0 && j%2!=0){
                count_rows=0;

                // 调用池化模块
                n_cycles_process_pooling = process_pooling_HCW(i, j, cols, layer_parameters);
                std::cout<<"process_pooling need cycles : "<<n_cycles_process_pooling<<std::endl;

                // 将池化结果写入DRAM
                n_cycles_write_output = store_pooling_output_HCW(ofmap, dram_instance, i, j, cols, layer_parameters);
                //std::cout<<"write output need cycles : "<<n_cycles_write_output<<std::endl;
            } else {
                n_cycles_process_pooling = 0;
                n_cycles_write_output = 0;
            }

            this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_process_pooling);
            std::cout<<"The current input loop calculation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
            std::cout<<std::endl;

            std::cout<<"write neuron_state need cycles : "<<n_cycles_write_neuron_state<<std::endl;
            std::cout<<"write output need cycles : "<<n_cycles_write_output<<std::endl;
            std::cout<<"write all result need cycles : "<<(n_cycles_write_output + n_cycles_write_neuron_state)<<std::endl;

             // 加上最后一次写数据所需的周期
            if(i==num_weight_buffer_fold-1 && j==X_-1){
                this->n_cycles += n_cycles_write_output;
                this->n_cycles += n_cycles_write_neuron_state;
                std::cout<<"Write the last output result to DRAM, and the global cycles is : "<<this->n_cycles<<std::endl;
            } else {
                std::cout<<"No need to add the number of cycles required for writing, and the global cycles is : "<<this->n_cycles<<std::endl;
            }

            PingPongBuffer_Switch(this->ppbuf_bank, this->input_pp_bank_0, this->input_pp_bank_1);
        }

        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); 
        delete[] this->on_chip_sram;
    }

    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete this->ppbuf_weight;

    delete this->ppbuf_bank;
    delete[] this->input_base_bank_0;
    delete[] this->input_base_bank_1;
    delete[] this->input_pp_bank_0;
    delete[] this->input_pp_bank_1;

    delete[] this->output_buffer;
    delete[] this->output_buffer_cpu;
    delete[] this->neuron_state_buffer;

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_and_pooling_compute_HCW(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查神经元状态
    for(int i=0; i<X_*Y_*K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

        // 检查输出
    for(int i=0; i<X_*Y_*K/4; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<"The current layer simulation ends, and the global cycles is : "<<this->n_cycles<<std::endl;
    std::cout<<"             \033[1;32m"<<"Test passed correctly"<<"\033[0m"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;

    // std::cout<<"below is ofmap data : "<<std::endl;
    // for(int channels=0; channels<K; channels++){
    //     std::cout<<"channels : "<<channels<<std::endl;
    //     for(int p=0; p<X_/2; p++){
    //         for(int q=0; q<Y_/2; q++){
    //             // int index = p * Y_/2 * K + q * K + channels;  // HWC layout
    //             int index = p * K * Y_/2 + channels * Y_/2 + q; // HCW layout
    //             std::cout<<ofmap[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}


std::tuple<int*, int*, int*, int*> Controller::runConv(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m       "<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } else {
        // 将输入数据和基地址和输出数据的基地址交换
        uint64_t temp = this->input_offset;
        this->input_offset = this->output_offset;
        this->output_offset = temp;
    }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            // std::cout<<"i : "<<i<<std::endl;
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
            // std::cout<<r.num_rows<<std::endl;
            // std::cout<<r.start_rows<<std::endl;
            // std::cout<<r.add_0_above<<std::endl;
            // std::cout<<r.add_0_below<<std::endl;
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    std::cout<<"below is ifmap data : "<<std::endl;
    for(int channels=0; channels<C; channels++){
        std::cout<<"channels : "<<channels<<std::endl;
        for(int p=0; p<X; p++){
            for(int q=0; q<Y; q++){
                // int index = p * Y * C + q * C + channels;  // HWC layout
                int index = p * C * Y + channels * Y + q; // HCW layout
                std::cout<<ifmap[index]<<"  ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    std::cout<<"------------------------------------------"<<std::endl;

    int num_input_buffer_need = R * Y_padded * C;  // 加padding
    assert(num_input_buffer_need <= this->num_input);
    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    // 第一次加载input data
    // 加载数据到用于计算的buffer，加载到next_buffer
    int n_cycles_load_first_input = load_input_data_HCW(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
    // 切换，切换之后，上一步加载的数据到了current_buffer，用于下面的计算，next_buffer是空的，用于加载下一块数据（和计算同时进行）
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
    this->n_cycles += n_cycles_load_first_input;
    std::cout<<"load the first input cycles : "<<n_cycles_load_first_input<<std::endl;

    for(int i=0; i<X_; i++){
        std::cout<<"     i = "<<i<<std::endl;

        std::cout<<"below is input_buffer data : "<<std::endl;

        for(int channels=0; channels<C; channels++){
            std::cout<<"channels : "<<channels<<std::endl;
            for(int p=0; p<R; p++){
                for(int q=0; q<Y_padded; q++){
                    // int index = p * Y_padded * C + q * C + channels;  // HWC layout
                    int index = p * C * Y_padded + channels * Y_padded + q; // HCW layout
                    std::cout<<this->ppbuf_input->current_buffer[index]<<"  ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;

        // 调用im2col函数
        for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
            int start = num_tile*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            int cycles_im2col = im2col_HCW(start, num, layer_parameters);
        }

        if(i+1<X_){
            int n_cycles_load_next_input = load_input_data_HCW(layer_id, ifmap, this->dram_instance, i+1, layer_parameters);  // 调用函数**************************
            //this->n_cycles += n_cycles_load_input;
            std::cout<<"load next input cycles : "<<n_cycles_load_next_input<<std::endl;
            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        }
    }

    delete this->ppbuf_input;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;

    return {ifmap, filter, ofmap, nfmap};
}