import math
import sys
import torch
from torch._utils import _get_device_index as _torch_get_device_index


def available_gpu(need_mem: int, need_cap: float=0.0) -> tuple[bool, list]:

    ids: list = list()
    is_available_gpu: bool = torch.cuda.is_available()

    if is_available_gpu:
        
        info: dict = get_gpu_info()
        ids: list = list()
        remaining_memories: list = list()

        for i in info.keys():

            info_i: dict= info.get(i)
            compute_capability: float = info_i.get("compute_capability")
            memory: dict = info_i.get("memory")
            remaining_memory: int = memory.get("remaining_memory")
            is_over_than_need_cap: bool = need_cap <= compute_capability
            is_over_than_need_mem: bool = need_mem <= remaining_memory

            if is_over_than_need_cap and is_over_than_need_mem:
                
                ids.append(i)
                remaining_memories.append(remaining_memory)

        indices: list = sorted(
            range(len(remaining_memories)),
            key=lambda k: remaining_memories[k],
        )
        ids: list = [ids[i] for i in indices]
        remaining_memories: list = [remaining_memories[i] for i in indices]

    return is_available_gpu, ids

def get_gpu_info() -> dict:

    info: dict = {}
    gpu_num: int = torch.cuda.device_count()
    
    for i in range(gpu_num):

        try:

            compute_capability: float = float(
                ".".join(map(str, torch.cuda.get_device_capability(i)))
            )
            remaining_memory, total_memory = torch.cuda.mem_get_info(i)
            info[str(i)] = {
                "compute_capability": compute_capability,
                "memory": {
                    "remaining_memory": remaining_memory,
                    "total_memory": total_memory,
                },
            }

        # Invalid device id
        except AssertionError as e:

            pass

    print(f"{sys._getframe(0).f_code.co_name} - Info : {info}.")

    return info