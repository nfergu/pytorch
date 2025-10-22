#include <c10/core/impl/alloc_cpu.h>
#include <c10/mobile/CPUCachingAllocator.h>
#include <c10/util/Exception.h>

namespace c10 {

namespace {
thread_local CPUCachingAllocator* caching_allocator_ptr{nullptr};
} // namespace

std::mutex CPUCachingAllocator::mutex_;
ska::flat_hash_map<void*, size_t> CPUCachingAllocator::allocation_map_;

inline void* CPUCachingAllocator::allocate_and_cache(const size_t bytes) {
  void* ptr = nullptr;
  try {
    ptr = c10::alloc_cpu(bytes);
  } catch (c10::Error&) {
    // If allocation fails, try freeing cached available blocks.
    // For now free all available cached blocks.
    free_cached();
    // Furthermore to consider: If we ever come here running out of memory
    // perhaps it is best to disable caching, since this is likely to happen
    // again.
    // Try again.
    ptr = c10::alloc_cpu(bytes);
  }
  allocation_map_[ptr] = bytes;
  return ptr;
}

void* CPUCachingAllocator::allocate(const size_t bytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto& it = available_map_.find(bytes);
  if (it == available_map_.end() || it->second.empty()) {
    return allocate_and_cache(bytes);
  }
  void* ptr = it->second.pop_back_val();
  // Update cached bytes count
  total_cached_bytes_ -= bytes;
  return ptr;
}

void CPUCachingAllocator::free_cached_memory_if_needed(const size_t bytes) {
  // If adding this allocation would exceed the cache limit, free some memory
  // We use a simple FIFO strategy: free cached blocks until we have enough space
  if (total_cached_bytes_ + bytes <= max_cached_bytes_) {
    return;
  }
  
  // Free cached blocks starting with the largest sizes (most likely to be
  // unused) until we have enough space
  while (total_cached_bytes_ + bytes > max_cached_bytes_ && !available_map_.empty()) {
    // Find the largest cached size
    size_t max_size = 0;
    for (const auto& it : available_map_) {
      if (!it.second.empty() && it.first > max_size) {
        max_size = it.first;
      }
    }
    
    if (max_size == 0) {
      break;  // No more blocks to free
    }
    
    // Free one block of the largest size
    auto& blocks = available_map_[max_size];
    if (!blocks.empty()) {
      void* ptr = blocks.pop_back_val();
      c10::free_cpu(ptr);
      allocation_map_.erase(ptr);
      total_cached_bytes_ -= max_size;
    }
    
    // Remove the entry if it's now empty
    if (blocks.empty()) {
      available_map_.erase(max_size);
    }
  }
}

void CPUCachingAllocator::free(void* ptr) {
  // NB: since we are not really freeing the memory
  // the cases such as quantization code freeing original weights
  // on mobile, will not quite work, as we likely will hold
  // onto that memory.
  // NB: We can also enable max memory cached for better memory
  // management such that free will actually free the memory if
  // we are nearing or above the watermark.
  std::lock_guard<std::mutex> guard(mutex_);
  // If this allocation was done before caching allocator was enabled
  // then free regularly
  const auto& it = allocation_map_.find(ptr);
  if (it == allocation_map_.end()) {
    c10::free_cpu(ptr);
    return;
  }
  const size_t alloc_size = it->second;
  
  // Check if we need to free some cached memory before caching this block
  free_cached_memory_if_needed(alloc_size);
  
  // If after freeing we still don't have space, just free this block directly
  if (total_cached_bytes_ + alloc_size > max_cached_bytes_) {
    c10::free_cpu(ptr);
    allocation_map_.erase(ptr);
    return;
  }
  
  // Cache the block
  available_map_[alloc_size].push_back(ptr);
  total_cached_bytes_ += alloc_size;
}

void CPUCachingAllocator::record_free(void* ptr) {
  // This function captures the case when the allocated memory
  // is being freed outside the scope of this allocator.
  // At the moment only way to capture this is to have the allocator,
  // that uses this CachingAllocator as the backing allocator,
  // call this function explicitly upon freeing memory while
  // outside the scope of caching allocator.
  // If the memory is freed in some other way, then we will likely
  // have undefined behavior or page fault. But this can be
  // the case without caching allocator as well.
  std::lock_guard<std::mutex> guard(mutex_);
  const auto& it = allocation_map_.find(ptr);
  if (it != allocation_map_.end()) {
    allocation_map_.erase(it);
  }
}

void CPUCachingAllocator::free_cached() {
  for (const auto& it : available_map_) {
    for (const auto ptr : it.second) {
      c10::free_cpu(ptr);
      // When cached memory is return to OS, it must be removed
      // from allocation_map.
      allocation_map_.erase(ptr);
    }
  }
  available_map_.clear();
  total_cached_bytes_ = 0;
}

CPUCachingAllocator::~CPUCachingAllocator() {
  free_cached();
}

CPUCachingAllocator* GetThreadLocalCachingAllocator() {
  return caching_allocator_ptr;
}

WithCPUCachingAllocatorGuard::WithCPUCachingAllocatorGuard(
    CPUCachingAllocator* allocator)
    : prev_caching_allocator_ptr_(GetThreadLocalCachingAllocator()) {
  caching_allocator_ptr = allocator;
}

WithCPUCachingAllocatorGuard::~WithCPUCachingAllocatorGuard() {
  caching_allocator_ptr = prev_caching_allocator_ptr_;
}

} // namespace c10
