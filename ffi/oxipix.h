#ifndef OXIPIX_H
#define OXIPIX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OXIPIX_FORMAT_JPEG = 0,
    OXIPIX_FORMAT_WEBP = 1,
    OXIPIX_FORMAT_PNG = 2,
    OXIPIX_FORMAT_JXL = 3
} oxipix_output_format;

typedef struct {
    uint32_t region_x;
    uint32_t region_y;
    uint32_t region_w;
    uint32_t region_h;
    uint32_t out_w;
    uint32_t out_h;
    uint8_t quality;
    oxipix_output_format format;
} oxipix_request;

typedef struct {
    uint32_t width;
    uint32_t height;
} oxipix_image_info;

int oxipix_init(uint64_t tile_cache_mb, uint64_t output_cache_mb);
int oxipix_get_info(const char *path, oxipix_image_info *info, char **error_message);
int oxipix_process(const char *path, const oxipix_request *request, uint8_t **out_data, size_t *out_len, char **error_message);
void oxipix_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
