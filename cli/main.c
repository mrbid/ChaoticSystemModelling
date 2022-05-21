/*
    James William Fletcher (james@voxdsp.com)
        May 2022

    Info:

        This is the client version of dataset generation, multi-process.

        This produces the full simulation model, a neural network that
        feeds back its own output as input for the next step in the
        simulation. The theorys is that all that needs to be generated
        is the inital start states of the spheres.

        We can reduce the output vector by only outputting positions
        for each sphere and then inferring the direction by taking the
        old position from the new position and normalising the vector. 

        Saves parameters and we'd have had to normalise any unit vectors
        returned from the neural network anyway because it's unlikely
        they'd have been perfect.

        I really like this model because I can produce large unique
        datasets very quickly.
        
        This is configured to output 400,000 samples of data which is
        replayed at 60fps so that is 1.85 hours worth of data.
        
*/

#include <stdio.h>
#include <time.h>

#include <sys/file.h>
#include <stdint.h>
#include <unistd.h>

#include "../inc/vec.h"

#define f32 float

#ifndef __x86_64__
    #define NOSSE
#endif

#define SEIR_RAND

//*************************************
// globals
//*************************************
#define MAX_SPHERES 16
f32 SPHERE_SCALE = 0.16f;
f32 SPHERE_SPEED = 0.003f;

typedef struct{vec pos, dir;} sphere;
sphere spheres[MAX_SPHERES];

#define MAX_MEM_X 38400000 // 400000*6*16
#define MAX_MEM_Y 19200000 // MAX_MEM_X / 2
// (~200mb for both files) (~13.5gb with 64 processes) (~27gb with 128 processes)
f32 dataset_x[MAX_MEM_X];
f32 dataset_y[MAX_MEM_Y];
uint ix = 0, iy = 0;

//*************************************
// utility functions
//*************************************
void timestamp(char* ts)
{
    const time_t tt = time(0);
    strftime(ts, 16, "%H:%M:%S", localtime(&tt));
}

uint64_t urand()
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return s;
}

void writeWarning(const char* s)
{
    FILE* f = fopen("WARNING_FLAGGED_ERROR.TXT", "a"); // just make it long so that it is noticable
    if(f != NULL)
    {
        char strts[16];
        timestamp(&strts[0]);
        fprintf(f, "[%s] %s\n", strts, s);
        printf("[%s] %s\n", strts, s);
        fclose(f);
    }
}

void dumpBuffers()
{
    // open and lock the X file and don't unlock until Y is also written to
    int fx = open("dataset_x.dat", O_APPEND | O_CREAT | O_WRONLY, S_IRWXU);
    if(fx > -1)
    {
        // lock X
        if(flock(fx, LOCK_EX) == -1) // very rare that these would hang forever unless there is some serious hard drive failure.
            usleep(1000);

        // append to X file
        const size_t ixs = ix*sizeof(f32);
        const ssize_t wb = write(fx, &dataset_x[0], ixs);
        if(wb != ixs) // this is very rare but if it fails... well.. we have a log
        {
            char emsg[256];
            sprintf(emsg, "Just wrote corrupted bytes to X file! (last %zu bytes).", wb);
            writeWarning(emsg);
            exit(0);
        }

        // open Y file but we don't need to lock it as the X file lock is governing both
        int fy = open("dataset_y.dat", O_APPEND | O_CREAT | O_WRONLY, S_IRWXU);
        if(fy > -1)
        {
            // append to Y file
            const size_t iys = iy*sizeof(f32);
            const ssize_t wb = write(fy, &dataset_y[0], iys);
            if(wb != iys) // this is very rare but if it fails... well.. we have a log
            {
                char emsg[256];
                sprintf(emsg, "Just wrote corrupted bytes toY file! (last %zu bytes).", wb);
                writeWarning(emsg);
                exit(0);
            }

            // close Y
            close(fy);
        }
        else
        {
            writeWarning("Failed to open Y file.");
            exit(0);
        }

        // unlock X
        if(flock(fx, LOCK_UN) == -1)
            usleep(1000);

        // close X
        close(fx);
    }
}

//*************************************
// Process Entry Point
//*************************************
int main(int argc, char** argv)
{
    // init random starting state
    srandf(urand());
    for(uint i = 0; i < MAX_SPHERES; i++)
    {
        vRuvTA(&spheres[i].pos); // random point on inside of unit sphere
        vRuvBT(&spheres[i].dir); // random point on outside of unit sphere
        vNorm(&spheres[i].dir);
    }

    // pre-compute
    const f32 SPHERE_SCALE_2 = SPHERE_SCALE*2.f;
    
    // run full pelt until buffer is filled then dump it to a file and exit.
    while(1)
    {
        for(uint i = 0; i < MAX_SPHERES; i++)
        {
            dataset_x[ix++] = spheres[i].pos.x;
            dataset_x[ix++] = spheres[i].pos.y;
            dataset_x[ix++] = spheres[i].pos.z;
            dataset_x[ix++] = spheres[i].dir.x;
            dataset_x[ix++] = spheres[i].dir.y;
            dataset_x[ix++] = spheres[i].dir.z;

            vec inc;
            vMulS(&inc, spheres[i].dir, SPHERE_SPEED);
            vAdd(&spheres[i].pos, spheres[i].pos, inc);

            if(vMod(spheres[i].pos) > 1.f)
            {
                vec sd = spheres[i].pos;
                vNorm(&sd);

                vReflect(&spheres[i].dir, spheres[i].dir, sd);
                vNorm(&spheres[i].dir);
            }

            for(uint j = 0; j < MAX_SPHERES; j++)
            {
                if(j == i){continue;} // dont collide with self

                if(vDist(spheres[i].pos, spheres[j].pos) < SPHERE_SCALE_2)
                {
                    vReflect(&spheres[i].dir, spheres[j].dir, spheres[i].dir);
                    vNorm(&spheres[i].dir);
                }
            }

            dataset_y[iy++] = spheres[i].pos.x;
            dataset_y[iy++] = spheres[i].pos.y;
            dataset_y[iy++] = spheres[i].pos.z;
            // dataset_y[iy++] = spheres[i].dir.x;
            // dataset_y[iy++] = spheres[i].dir.y;
            // dataset_y[iy++] = spheres[i].dir.z;
            
            if(iy >= MAX_MEM_Y)
            {
                // dump buffers and quit
                dumpBuffers();
                return 0;
            }
        }
    }

    // done
    return 0;
}
