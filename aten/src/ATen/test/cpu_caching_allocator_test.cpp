#include <gtest/gtest.h>

#include <ATen/cpu/vec/vec.h>
#include <ATen/ATen.h>

#include <c10/mobile/CPUCachingAllocator.h>

// At the moment caching allocator is only exposed to mobile cpu allocator.
#ifdef C10_MOBILE

TEST(CPUCachingAllocatorTest, check_alloc_free) {
  c10::CPUCachingAllocator caching_allocator;
  c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
      &caching_allocator);
  at::Tensor a = at::rand({23, 23});
  float* data_ptr = a.data_ptr<float>();
  a.reset();
  a = at::rand({23, 23});
  ASSERT_TRUE(data_ptr == a.data_ptr<float>());
}

// This should just free the pointer correctly.
TEST(CPUCachingAllocatorTest, check_alloc_outside_free_inside) {
  c10::CPUCachingAllocator caching_allocator;
  at::Tensor a = at::rand({23, 23});
  {
    c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
        &caching_allocator);
    [[maybe_unused]] float* data_ptr = a.data_ptr<float>();
    a.reset();
    a = at::rand({23, 23});
  }
}

TEST(CPUCachingAllocatorTest, check_alloc_inside_free_outside) {
  c10::CPUCachingAllocator caching_allocator;
  at::Tensor a;
  {
    c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
        &caching_allocator);
    a = at::rand({23, 23});
  }
  a.reset();
}

TEST(CPUCachingAllocatorTest, check_cache_size_limit) {
  c10::CPUCachingAllocator caching_allocator;
  // Set a small cache limit for testing (100KB)
  caching_allocator.set_max_cached_bytes(100 * 1024);
  
  c10::WithCPUCachingAllocatorGuard cachine_allocator_guard(
      &caching_allocator);
  
  // Allocate and free a large tensor (should exceed cache limit)
  // Each float is 4 bytes, so 500x500 = 1MB
  {
    at::Tensor large = at::rand({500, 500});
    // Free it by letting it go out of scope
  }
  
  // The large tensor should not be fully cached due to size limit
  size_t cached_after_large = caching_allocator.get_cached_bytes();
  ASSERT_LE(cached_after_large, 100 * 1024);
  
  // Allocate and free multiple small tensors
  for (int i = 0; i < 10; i++) {
    at::Tensor small = at::rand({10, 10});
    // Free it by letting it go out of scope
  }
  
  // Cache should still be under limit
  size_t cached_after_small = caching_allocator.get_cached_bytes();
  ASSERT_LE(cached_after_small, 100 * 1024);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  at::manual_seed(42);
  return RUN_ALL_TESTS();
}

#endif /* C10_Mobile */
