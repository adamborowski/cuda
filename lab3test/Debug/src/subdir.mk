################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/lab3test.cu 

CU_DEPS += \
./src/lab3test.d 

OBJS += \
./src/lab3test.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -I/usr/local/cuda/samples/common/inc -G -g -lineinfo -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -m64 -I/usr/local/cuda/samples/common/inc -G -g -lineinfo -O0 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


