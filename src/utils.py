import os
import gc
import psutil
import torch
import GPUtil
import re
import logging
import numpy as np

def log_memory_usage(logger=None):
    cpu_memory_percent = psutil.virtual_memory().percent
    if logger:
        logger.info(f"Total RAM Used: {cpu_memory_percent}%")
    else:
        print(f"Total RAM Used: {cpu_memory_percent}%")
    
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if logger:
            logger.info(f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        else:
            print(f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal} MB")

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

def get_dynamic_batch_size(initial_batch_size=128, min_batch_size=32):
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        available_memory = total_memory - reserved_memory
        
        if available_memory < 10 * 1024 * 1024 * 1024:
            return max(initial_batch_size // 4, min_batch_size)
        elif available_memory < 20 * 1024 * 1024 * 1024:
            return initial_batch_size // 2
        return initial_batch_size
    except Exception:
        return initial_batch_size

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
