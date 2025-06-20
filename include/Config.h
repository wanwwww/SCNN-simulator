#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <iostream>
#include "types.h"

//--------------------------------------------------------------------
// DSNetework Configuration Parameters
//--------------------------------------------------------------------
// class DSNetworkConfig {
// public:
//     unsigned int n_switches_traversed_by_cycle; //TODO Not implemented yet

//     void printConfiguration(std::ofstream& out, unsigned int indent);
// };

//--------------------------------------------------------------------
// DSwitch Configuration Parameters 
//--------------------------------------------------------------------
// class DSwitchConfig {
// public:
//     //By the moment there is nothing to configure for the DSwitch
//     unsigned int latency; 
//     unsigned int input_ports; //Number of input_ports. By default this will be 1
//     unsigned int output_ports; //Number of output ports. By default this will be 2
//     unsigned int port_width; //Bit width

//     void printConfiguration(std::ofstream& out, unsigned int indent);
// };

//--------------------------------------------------------------------
// ADD
// DRAM configuration parameters
//--------------------------------------------------------------------
class DRAMConfig {
    public:
        uint64_t input_offset;
        uint64_t weight_offset;    
        uint64_t output_offset;
        uint64_t neuron_state_offset;
    
        void printConfiguration(std::ofstream& out, unsigned int indent);
};

//--------------------------------------------------------------------
// ADD
// On-chip buffer configuration parameters
//--------------------------------------------------------------------
class BufferConfig {
public:
    unsigned int input_buffer_size;   // KB
    unsigned int weight_buffer_size;  
    unsigned int neuron_state_buffer_size;  
    unsigned int output_buffer_size;

    void printConfiguration(std::ofstream& out, unsigned int indent);
};


//--------------------------------------------------------------------
// MSNetwork Configuration Parameters
//--------------------------------------------------------------------
class MSNetworkConfig {
public:
    //MultiplierNetwork_t multiplier_network_type;
    //unsigned int ms_size; //Number of multiplier switches. 
    unsigned int ms_rows;
    unsigned int ms_cols; 

    void printConfiguration(std::ofstream& out, unsigned int indent);
};

//--------------------------------------------------------------------
// MSwitch Configuration Parameters
//--------------------------------------------------------------------
class MSwitchConfig {
public:
    cycles_t latency; //Latency of the MS to perform a multiplication. This number is expressed in number of cycles. //TODO To imple
    unsigned int input_ports; //Number of input ports of the MS. This number is 1 by default in MAERI
    unsigned int output_ports; // Number of output ports of the MS. 
    unsigned int forwarding_ports; // Number of forwarding ports of the MS. This is basically the number of elements that can be forwarded in a single cycle and in MAERI architecture is just 1.
    unsigned int port_width; //Bit width 
    unsigned int buffers_capacity; //Number of elements that can be stored in the MS buffers. TODO In future implementations this could be splited up, taking each buffer capacity in a different parameter.

    void printConfiguration(std::ofstream& out, unsigned int indent);
};

//--------------------------------------------------------------------
// ASNetwork Configuration Parameters
//--------------------------------------------------------------------
class ASNetworkConfig {
public:
    //ReduceNetwork_t reduce_network_type; //Type of the ReduceNetwork configured in this moment
    unsigned int accumulation_buffer_enabled;  
    void printConfiguration(std::ofstream& out, unsigned int indent);
};

//--------------------------------------------------------------------
// ASwitch Configuration Parameters
//--------------------------------------------------------------------
class ASwitchConfig {
public:
    unsigned int buffers_capacity;
    unsigned int input_ports; //Number of input ports of the ASwitch. By  default in MAERI this is just 2
    unsigned int output_ports; //Number of output ports of the ASwitch. By default  in MAERI this is 1
    unsigned int forwarding_ports; //Nuber of forwarding ports of the ASwitch.
    unsigned int port_width; //Bit width
    cycles_t latency; //Latency of the AS to perform. This number is expressed in number of cycles. //TODO To implement

    void printConfiguration(std::ofstream& out, unsigned int indent);
};

//--------------------------------------------------------------------
// LookUpTable Configuration Parameters
//--------------------------------------------------------------------
// class LookUpTableConfig {
// public:
//     cycles_t latency; //Latency of the LookUpTable to perform. This number must be expressed in number of cycles. 0 no supported
//     unsigned int port_width; 

//     void printConfiguration(std::ofstream& out, unsigned int indent);

// };


//--------------------------------------------------------------------
// UpdateNetwork Configuration Parameters
//--------------------------------------------------------------------
class UpdateNetworkConfig {
public:
    void printConfiguration(std::ofstream& out, unsigned int indent);  
};

//--------------------------------------------------------------------
// UpdateSwitch Configuration Parameters
//--------------------------------------------------------------------
class UpdateSwitchConfig {
public:
    unsigned int port_width; //Bit width
    unsigned int buffers_capacity;
    cycles_t latency; //Latency of the AS to perform. This number is expressed in number of cycles. //TODO To implement
    void printConfiguration(std::ofstream& out, unsigned int indent);
};

//--------------------------------------------------------------------
// SDMemory Controller Configuration Parameters
//--------------------------------------------------------------------
class SDMemoryConfig {
public:
    //MemoryController_t mem_controller_type; //Memory controller type (e.g., DENSE_WORKLOAD or SPARSE_GEMM)
    unsigned int write_buffer_capacity; //Capacity of the buffers expressed in terms of number of elements
    unsigned int n_read_ports; //dn_bw
    unsigned int n_write_ports;  //rn_bw
    unsigned int port_width; //Bit width
    //unsigned int n_multiplier_configurations; //Number of multiplier configurations
    //unsigned int n_reduce_network_configurations; //Number of reduce network configurations

    void printConfiguration(std::ofstream& out, unsigned int indent);

};

//--------------------------------------------------------------------
//--------------------------------------------------------------------
//MAIN CONFIGURATION OBJECT
//--------------------------------------------------------------------
//--------------------------------------------------------------------

class Config {
public:

    // 特征图在DRAM中的存储顺序
    std::string storage_type;

    // SNN parameter
    data_t V_th;
    data_t Timestamp;
    Layer_t layer_type;
    pooling_t pooling_type;

    // Network weight parameters
    int weight_width;
    int max_weight;
    int min_weight;

    //General parameters
    bool print_stats_enabled;    //Specified whether the statistics must be printed. 
    
    //DSNetwork Configuration
    //DSNetworkConfig m_DSNetworkCfg;
    
    //DSwitch Configuration
    //DSwitchConfig m_DSwitchCfg;

    // DRAM configuration
    DRAMConfig m_DRAMCfg;

    // On-chip buffer configuration
    BufferConfig m_BufferCfg;

    //MSNetwork Configuration
    MSNetworkConfig m_MSNetworkCfg;

    //MSwitch Configuration
    MSwitchConfig m_MSwitchCfg;

    //ASNetwork Configuration
    ASNetworkConfig m_ASNetworkCfg;

    //ASwitch Configuration
    ASwitchConfig m_ASwitchCfg;

    //LookUpTableConfiguration
    //LookUpTableConfig m_LookUpTableCfg;

    //UpdateNetwork Configuration
    UpdateNetworkConfig m_UpdateNetworkCfg;

    //UpdateSwitch Configuration
    UpdateSwitchConfig m_UpdateSwitchCfg;

    //SDMemory controller configuration
    SDMemoryConfig m_SDMemoryCfg;

    //Constructor runs reset()
    Config();   // 构造函数运行reset()方法 

    //Load parameters from configuration file using TOML Syntax
    void loadFile(std::string config_file);

    //Reset parameters with default values
    void reset();

    //print the configuration parameters
    void printConfiguration(std::ofstream& out, unsigned int indent);

    // //Indicates whether according to the hardware parameters, sparsity is enabled in the architecture
    // bool sparsitySupportEnabled(); 

    // //Indicates whether according to the hardware parameters, the operation of CONV itself is supported. Otherwise, the operation can be done
    // //using GEMM operation
    // bool convOperationSupported(); 
};



#endif
