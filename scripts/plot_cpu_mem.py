from turtle import color
import numpy as np
import matplotlib.pyplot as plt



#---------------------------------------------------LIO-LIVOX------------------------------------------
CPU1_Data = np.load('./usage/kiss_cpu.npy')
CPU2_Data = np.load('./usage/kp_cpu.npy')
MEM1_Data = np.load('./usage/kiss_mem.npy')
MEM2_Data = np.load('./usage/kp_mem.npy')
CPU = CPU1_Data[4:] + CPU2_Data[4:]
MEM = MEM1_Data[4:] + MEM2_Data[4:]
print(CPU1_Data,CPU2_Data)
print(np.mean(CPU))
print(np.mean(MEM)/(1024*1024))
print("kiss_icp:", np.mean(CPU1_Data[4:]),np.mean(MEM1_Data[4:])/(1024*1024))

# # plt.plot(CPU3_Data)
plt.plot(CPU1_Data[4:],color='r')
plt.plot(CPU2_Data[4:],color='b')
# plt.plot(MEM1_Data,color='g')
# plt.plot(MEM2_Data)
plt.show()


#---------------------------------------------------ORIGINAL------------------------------------------

# CPU1_Data = np.load('./usage/kiss_cpu.npy')
# MEM1_Data = np.load('./usage/kiss_mem.npy')
# CPU = CPU1_Data[5:]
# MEM = MEM1_Data[5:]
# print(np.mean(CPU))
# print(np.mean(MEM)/(1024*1024))