#include <sys/mman.h>
#include <pthread.h>
#include <dlfcn.h>
#include <time.h>
#include <semaphore.h>
#include <signal.h>
#include <stddef.h>
#include <sys/stat.h>

#define GLIBC_VERSION "GLIBC_2.2.5"

extern "C" {

// This variable is only added at GLIBC_2.32,
// but only v2.31 is available in our execution environments.
// So we redefine that variable here to workaround that issue.
char __wrap___libc_single_threaded = 0;

__asm__(".symver shm_unlink,shm_unlink@" GLIBC_VERSION);
__asm__(".symver shm_open,shm_open@" GLIBC_VERSION);
__asm__(".symver pthread_join,pthread_join@" GLIBC_VERSION);
__asm__(".symver dlerror,dlerror@" GLIBC_VERSION);
__asm__(".symver __pthread_key_create,__pthread_key_create@" GLIBC_VERSION);
__asm__(".symver pthread_mutexattr_destroy,pthread_mutexattr_destroy@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_tryrdlock,pthread_rwlock_tryrdlock@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_destroy,pthread_rwlock_destroy@" GLIBC_VERSION);
__asm__(".symver pthread_setspecific,pthread_setspecific@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_timedwrlock,pthread_rwlock_timedwrlock@" GLIBC_VERSION);
__asm__(".symver sem_destroy,sem_destroy@" GLIBC_VERSION);
__asm__(".symver sem_wait,sem_wait@" GLIBC_VERSION);
__asm__(".symver pthread_once,pthread_once@" GLIBC_VERSION);
__asm__(".symver pthread_create,pthread_create@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_rdlock,pthread_rwlock_rdlock@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_timedrdlock,pthread_rwlock_timedrdlock@" GLIBC_VERSION);
__asm__(".symver pthread_kill,pthread_kill@" GLIBC_VERSION);
__asm__(".symver pthread_mutexattr_init,pthread_mutexattr_init@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_init,pthread_rwlock_init@" GLIBC_VERSION);
__asm__(".symver pthread_key_create,pthread_key_create@" GLIBC_VERSION);
__asm__(".symver pthread_rwlockattr_init,pthread_rwlockattr_init@" GLIBC_VERSION);
__asm__(".symver dlopen,dlopen@" GLIBC_VERSION);
__asm__(".symver sem_init,sem_init@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_unlock,pthread_rwlock_unlock@" GLIBC_VERSION);
__asm__(".symver pthread_detach,pthread_detach@" GLIBC_VERSION);
__asm__(".symver pthread_mutexattr_setpshared,pthread_mutexattr_setpshared@" GLIBC_VERSION);
__asm__(".symver sem_timedwait,sem_timedwait@" GLIBC_VERSION);
__asm__(".symver dladdr,dladdr@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_wrlock,pthread_rwlock_wrlock@" GLIBC_VERSION);
__asm__(".symver sem_post,sem_post@" GLIBC_VERSION);
__asm__(".symver pthread_rwlock_trywrlock,pthread_rwlock_trywrlock@" GLIBC_VERSION);
__asm__(".symver pthread_condattr_setpshared,pthread_condattr_setpshared@" GLIBC_VERSION);
__asm__(".symver sem_trywait,sem_trywait@" GLIBC_VERSION);
__asm__(".symver pthread_key_delete,pthread_key_delete@" GLIBC_VERSION);
__asm__(".symver pthread_getspecific,pthread_getspecific@" GLIBC_VERSION);
__asm__(".symver pthread_mutexattr_settype,pthread_mutexattr_settype@" GLIBC_VERSION);
__asm__(".symver dlvsym,dlvsym@" GLIBC_VERSION);
__asm__(".symver dlsym,dlsym@" GLIBC_VERSION);
__asm__(".symver pthread_mutex_trylock,pthread_mutex_trylock@" GLIBC_VERSION);
__asm__(".symver pthread_rwlockattr_setpshared,pthread_rwlockattr_setpshared@" GLIBC_VERSION);
__asm__(".symver pthread_rwlockattr_destroy,pthread_rwlockattr_destroy@" GLIBC_VERSION);
__asm__(".symver dlclose,dlclose@" GLIBC_VERSION);

int __wrap_pthread_join(pthread_t thread, void **value_ptr) {
    return pthread_join(thread, value_ptr);
}

char *__wrap_dlerror(void) {
    return dlerror();
}

int __wrap_pthread_mutexattr_destroy(pthread_mutexattr_t *attr) {
    return pthread_mutexattr_destroy(attr);
}

int __wrap_pthread_rwlock_tryrdlock(pthread_rwlock_t *rwlock) {
    return pthread_rwlock_tryrdlock(rwlock);
}

int __wrap_pthread_rwlock_destroy(pthread_rwlock_t *rwlock) {
    return pthread_rwlock_destroy(rwlock);
}

int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    return pthread_setspecific(key, value);
}

int __wrap_pthread_rwlock_timedwrlock (pthread_rwlock_t *__restrict rwlock, const struct timespec *__restrict abstime) {
    return pthread_rwlock_timedwrlock(rwlock, abstime);
}

int __wrap_sem_destroy(sem_t *sem) {
    return sem_destroy(sem);
}

int __wrap_sem_wait (sem_t *sem) {
    return sem_wait(sem);
}

int __wrap_pthread_once (pthread_once_t *once_control, void (*init_routine) (void)) {
    return pthread_once(once_control, init_routine);
}

int __wrap_pthread_create (pthread_t *__restrict thread, const pthread_attr_t *__restrict attr,
			   void *(*start_routine) (void *), void *__restrict arg) {
    return pthread_create(thread, attr, start_routine, arg);
}

int __wrap_pthread_rwlock_rdlock (pthread_rwlock_t *rwlock) {
    return pthread_rwlock_rdlock(rwlock);
}

int __wrap_pthread_rwlock_timedrdlock (pthread_rwlock_t *__restrict rwlock, const struct timespec *__restrict abstime) {
    return pthread_rwlock_timedrdlock(rwlock, abstime);
}

int __wrap_pthread_kill(pthread_t thread, int sig) {
    return pthread_kill(thread, sig);
}

int __wrap_pthread_mutexattr_init(pthread_mutexattr_t *attr) {
    return pthread_mutexattr_init(attr);
}

int __wrap_pthread_key_create(pthread_key_t *key, void (*destructor)(void*)) {
    return pthread_key_create(key, destructor);
}

int __wrap_pthread_rwlock_init(pthread_rwlock_t *__restrict rwlock, const pthread_rwlockattr_t *__restrict attr) {
    return pthread_rwlock_init(rwlock, attr);
}

int __wrap_pthread_rwlockattr_init(pthread_rwlockattr_t *attr) {
    return pthread_rwlockattr_init(attr);
}

void *__wrap_dlopen(const char *filename, int flags) {
    return dlopen(filename, flags);
}

int __wrap_sem_init(sem_t *sem, int pshared, unsigned int value) {
    return sem_init(sem, pshared, value);
}

int __wrap_pthread_rwlock_unlock(pthread_rwlock_t *rwlock) {
    return pthread_rwlock_unlock(rwlock);
}

int __wrap_pthread_detach(pthread_t thread) {
    return pthread_detach(thread);
}

int __wrap_pthread_mutexattr_setpshared(pthread_mutexattr_t *attr, int pshared) {
    return pthread_mutexattr_setpshared(attr, pshared);
}

int __wrap_sem_timedwait(sem_t *__restrict sem, const struct timespec *__restrict abstime) {
    return sem_timedwait(sem, abstime);
}

int __wrap_dladdr(const void *addr, Dl_info *info) {
    return dladdr(addr, info);
}

int __wrap_pthread_rwlock_wrlock(pthread_rwlock_t *rwlock) {
    return pthread_rwlock_wrlock(rwlock);
}

int __wrap_sem_post(sem_t *sem) {
    return sem_post(sem);
}

int __wrap_pthread_rwlock_trywrlock(pthread_rwlock_t *rwlock) {
    return pthread_rwlock_trywrlock(rwlock);
}

int __wrap_pthread_condattr_setpshared(pthread_condattr_t *attr, int pshared) {
    return pthread_condattr_setpshared(attr, pshared);
}

int __wrap_sem_trywait(sem_t *sem) {
    return sem_trywait(sem);
}

int __wrap_pthread_key_delete(pthread_key_t key) {
    return pthread_key_delete(key);
}

void *__wrap_pthread_getspecific(pthread_key_t key) {
    return pthread_getspecific(key);
}

int __wrap_pthread_mutexattr_settype(pthread_mutexattr_t *attr, int type) {
    return pthread_mutexattr_settype(attr, type);
}

void *__wrap_dlvsym(void *__restrict handle, const char *__restrict symbol,
                    const char *__restrict version) {
    return dlvsym(handle, symbol, version);
}

void *__wrap_dlsym(void *__restrict handle, const char *__restrict symbol) {
    return dlsym(handle, symbol);
}

int __wrap_pthread_mutex_trylock(pthread_mutex_t *mutex) {
    return pthread_mutex_trylock(mutex);
}

int __wrap_pthread_rwlockattr_setpshared(pthread_rwlockattr_t *attr, int pshared) {
    return pthread_rwlockattr_setpshared(attr, pshared);
}

int __wrap_pthread_rwlockattr_destroy(pthread_rwlockattr_t *attr) {
    return pthread_rwlockattr_destroy(attr);
}

int __wrap_dlclose(void *handle) {
    return dlclose(handle);
}

int __wrap_shm_unlink(const char *name) {
  return shm_unlink(name);
}

int __wrap_shm_open(const char *name, int oflag, mode_t mode) {
    return shm_open(name, oflag, mode);
}
}  // extern "C"
