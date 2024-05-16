#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    unsigned char *y;
    int frame_index;
    int width;
    int height;
} Frame;

Frame *load_y(FILE *file, int width, int height)
{
    Frame *frame = malloc(sizeof(Frame));
    frame->y = malloc((size_t)(width * height) * sizeof(unsigned char));

    fread(frame->y, 1, width * height, file);

    // Skip U and V components
    fseek(file, width * height / 2, SEEK_CUR);

    return frame;
}

Frame *process_yuv_frames(const char *file_path, int width, int height, int num_frames)
{
    FILE *file = fopen(file_path, "rb");
    if (file == NULL)
    {
        printf("Cannot open file.\n");
        return NULL;
    }

    Frame *frames = malloc(num_frames * sizeof(Frame));

    for (int i = 0; i < num_frames; i++)
    {
        Frame *frame = load_y(file, width, height);
        frame->frame_index = i;
        frame->width = width;
        frame->height = height;
        frames[i] = *frame;
        free(frame);
    }

    fclose(file);

    return frames;
}
int main()
{
    int width = 352;
    int height = 288;
    int num_frames = 40;
    Frame *frames = process_yuv_frames("foreman_cif-1.yuv", width, height, num_frames);

    if (frames != NULL)
    {
        for (int i = 0; i < num_frames; i++)
        {
            printf("First byte of Y frame %d: %d\n", i, frames[i].y[0]);
        }

        for (int i = 0; i < num_frames; i++)
        {
            free(frames[i].y);
        }
        free(frames);
    }

    return 0;
}