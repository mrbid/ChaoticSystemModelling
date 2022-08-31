/*
    James William Fletcher (github.com/mrbid)
        May 2022

    Info:
    
        This is a simulation of spheres bouncing around in a unit sphere.

        The goal is to train a Feed-forward Neural Network to produce this
        simulation in two respects;

****    1. To learn the entire simulation, inputting sphere position and direction
        on each timestep and returning the new position and direction.

        2. To learn only when collisions are taking place in the simulation
        and to return the new direction vector.

        The traditional simulation is referred to as "CPU", and the Neural Network as "AI".

        The CPU version will be used as a supervisor to grade the hit and miss rate of
        collisions simulated by the AI version.

        A goal is also to allow mixed collisions, some spheres CPU controlled and some AI
        in the same unit sphere simulation.


    Keys:

        N = New simulation.
        F = FPS to console.
        P = Toggle CPU and NEURAL modes.
        
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/file.h>
#include <stdint.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/time.h>

#define uint GLushort
#define sint GLshort
#define f32 GLfloat

#include "inc/gl.h"
#define GLFW_INCLUDE_NONE
#include "inc/glfw3.h"

#ifndef __x86_64__
    #define NOSSE
#endif

// uncommenting this define will enable the MMX random when using fRandFloat (it's a marginally slower)
#define SEIR_RAND

#include "inc/esAux2.h"

#include "inc/res.h"
#include "inc/low.h"

//*************************************
// globals
//*************************************
GLFWwindow* window;
uint winw = 1024;
uint winh = 768;
double t = 0;   // time
f32 dt = 0;     // delta time
double fc = 0;  // frame count
double lfct = 0;// last frame count time
double lc = 0;  // logic count
double llct = 0;// last logic count time
f32 aspect;
double x,y,lx,ly;
double rww, ww, rwh, wh, ww2, wh2;
double uw, uh, uw2, uh2; // normalised pixel dpi

// render state id's
GLint projection_id;
GLint modelview_id;
GLint position_id;
GLint lightpos_id;
GLint solidcolor_id;
GLint color_id;
GLint opacity_id;
GLint normal_id; // 

// render state matrices
mat projection;
mat view;
mat model;
mat modelview;

// render state inputs
vec lightpos = {0.f, 0.f, 0.f};

// models
ESModel mdlSphere;

// game vars
#define FAR_DISTANCE 1000.f
#define MAX_SPHERES  16
uint RENDER_PASS = 0;
double st=0; // start time
char tts[32];// time taken string

// camera vars
uint focus_cursor = 1;
double sens = 0.001f;
f32 xrot = 2.6f;
f32 yrot = 1.f;
f32 zoom = -3.3f;

// sim
f32 SPHERE_SCALE = 0.16f;
f32 SPHERE_SPEED = 0.003f;

typedef struct{vec pos, dir; unsigned char c;} sphere;
sphere spheres[MAX_SPHERES];

uint neural_sim = 0;

//*************************************
// utility functions
//*************************************
void timestamp(char* ts)
{
    const time_t tt = time(0);
    strftime(ts, 16, "%H:%M:%S", localtime(&tt));
}

static inline float fRandFloat(const float min, const float max)
{
    return min + randf() * (max-min); 
}

void timeTaken(uint ss)
{
    if(ss == 1)
    {
        const double tt = t-st;
        if(tt < 60.0)
            sprintf(tts, "%.0f Sec", tt);
        else if(tt < 3600.0)
            sprintf(tts, "%.2f Min", tt * 0.016666667);
        else if(tt < 216000.0)
            sprintf(tts, "%.2f Hr", tt * 0.000277778);
        else if(tt < 12960000.0)
            sprintf(tts, "%.2f Days", tt * 0.00000463);
    }
    else
    {
        const double tt = t-st;
        if(tt < 60.0)
            sprintf(tts, "%.0f Seconds", tt);
        else if(tt < 3600.0)
            sprintf(tts, "%.2f Minutes", tt * 0.016666667);
        else if(tt < 216000.0)
            sprintf(tts, "%.2f Hours", tt * 0.000277778);
        else if(tt < 12960000.0)
            sprintf(tts, "%.2f Days", tt * 0.00000463);
    }
}

static inline uint isnorm(const float f)
{
    if(isnormal(f) == 1 || f == 0.f)
        return 1;
    return 0;
}

float urandf()
{
    static const float FLOAT_UINT64_MAX = (float)UINT64_MAX;
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return (((float)s)+1e-7f) / FLOAT_UINT64_MAX;
}

float uRandFloat(const float min, const float max)
{
    return ( urandf() * (max-min) ) + min;
}

uint64_t urand()
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return s;
}

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

//*************************************
// render functions
//*************************************

__attribute__((always_inline)) inline void modelBind(const ESModel* mdl) // C code reduction helper (more inline opcodes)
{
    glBindBuffer(GL_ARRAY_BUFFER, mdl->vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdl->iid);
}

void rSphere(f32 x, f32 y, f32 z)
{
    mIdent(&model);
    mTranslate(&model, x, y, z);
    mScale(&model, SPHERE_SCALE, SPHERE_SCALE, SPHERE_SCALE);
    mMul(&modelview, &model, &view);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);

    glDrawElements(GL_TRIANGLES, low_numind, GL_UNSIGNED_SHORT, 0);
}

//*************************************
// game functions
//*************************************

void newSim()
{
    const int seed = urand();
    srand(seed);
    srandf(seed);

    st = t;

    char strts[16];
    timestamp(&strts[0]);
    printf("[%s] Sim Start [%u].\n", strts, seed);
    
    glfwSetWindowTitle(window, "UnitCollider");
    
    for(uint i = 0; i < MAX_SPHERES; i++)
    {
        vRuvTA(&spheres[i].pos); // random point on inside of unit sphere
        vRuvBT(&spheres[i].dir); // random point on outside of unit sphere
        vNorm(&spheres[i].dir); // hmm or not?
        spheres[i].c = 0.f;
    }
}

//*************************************
// update & render
//*************************************
void main_loop()
{
//*************************************
// update title bar stats
//*************************************
    static double ltut = 3.0;
    if(t > ltut)
    {
        timeTaken(1);
        char title[512];
        if(neural_sim == 0)
            sprintf(title, "| %s | CPU", tts);
        else
            sprintf(title, "| %s | NEURAL", tts);
        glfwSetWindowTitle(window, title);
        ltut = t + 1.0;
    }

//*************************************
// camera
//*************************************

    if(focus_cursor == 1)
    {
        glfwGetCursorPos(window, &x, &y);

        xrot += (ww2-x)*sens;
        yrot += (wh2-y)*sens;

        if(yrot > 1.5f)
            yrot = 1.5f;
        if(yrot < 0.5f)
            yrot = 0.5f;

        glfwSetCursorPos(window, ww2, wh2);
    }

    mIdent(&view);
    mTranslate(&view, 0.f, 0.f, zoom);
    mRotate(&view, yrot, 1.f, 0.f, 0.f);
    mRotate(&view, xrot, 0.f, 0.f, 1.f);

//*************************************
// neural sim
//*************************************

    if(neural_sim == 1)
    {
        // write input to file
        FILE *f = fopen("/dev/shm/uc_input.dat", "wb");
        if(f != NULL)
        {
            float input[96];
            for(uint i = 0; i < MAX_SPHERES; i++)
            {
                const uint ofs = i * 6;
                input[ofs]   = spheres[i].pos.x;
                input[ofs+1] = spheres[i].pos.y;
                input[ofs+2] = spheres[i].pos.z;
                input[ofs+3] = spheres[i].dir.x;
                input[ofs+4] = spheres[i].dir.y;
                input[ofs+5] = spheres[i].dir.z;
            }
            if(fwrite(&input[0], sizeof(float), 96, f) != 96)
                printf("ERROR: neural write failed.\n");
            fclose(f);
        }

        // load last result
        float ret[48];
        f = fopen("/dev/shm/uc_r.dat", "rb");
        if(f != NULL)
        {
            if(fread(&ret, sizeof(float), 48, f) == 48)
            {
                // cant just dump the buffer, need to check norm and generate directions
                for(uint i = 0; i < MAX_SPHERES; i++)
                {
                    const uint ofs = i*3;
                    if(isnorm(ret[ofs]) == 0 || isnorm(ret[ofs+1]) == 0 || isnorm(ret[ofs+2]) == 0){continue;}

                    spheres[i].c = 0;

                    // new pos
                    vec np;
                    np.x = ret[ofs];
                    np.y = ret[ofs+1];
                    np.z = ret[ofs+2];
                    
                    // new direction
                    vec nd;
                    vSub(&nd, np, spheres[i].pos);
                    vNorm(&nd);

                    //printf("%f\n", vDot(spheres[i].dir, nd));
                    if(fabsf(vDot(spheres[i].dir, nd)) < 0.9f){spheres[i].c = 1;}

                    // set new dir
                    spheres[i].dir = nd;

                    // increment by new dir
                    vec inc;
                    vMulS(&inc, spheres[i].dir, SPHERE_SPEED);
                    vAdd(&spheres[i].pos, spheres[i].pos, inc);
                }
            }
            fclose(f);
        }
    }

//*************************************
// render
//*************************************
    if(RENDER_PASS == 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    for(uint i = 0; i < MAX_SPHERES; i++)
    {
        if(RENDER_PASS == 1)
        {
            if(spheres[i].c == 0)
                glUniform3f(color_id, 0.f, 0.f, 1.f);
            else
                glUniform3f(color_id, 1.f, 0.f, 0.f);
        }

        if(neural_sim == 0)
        {
            vec inc;
            vMulS(&inc, spheres[i].dir, SPHERE_SPEED);
            vAdd(&spheres[i].pos, spheres[i].pos, inc);

            const f32 mod = vMod(spheres[i].pos);
            if(mod > 1.f)
            {
                vec sd = spheres[i].pos;
                vNorm(&sd);

                vReflect(&spheres[i].dir, spheres[i].dir, sd);
                vNorm(&spheres[i].dir); // hmm or not?

                vec ob = spheres[i].pos;
                vNorm(&ob);
                vInv(&ob);

                vec inc;
                vMulS(&inc, ob, (mod-1.f)+SPHERE_SPEED);
                // vMulS(&inc, spheres[i].dir, (mod-1.f)+SPHERE_SPEED);
                vAdd(&spheres[i].pos, spheres[i].pos, inc);
            }

            for(uint j = 0; j < MAX_SPHERES; j++)
            {
                if(j == i){continue;} // dont collide with self

                const f32 d = vDist(spheres[i].pos, spheres[j].pos);
                const f32 cd = SPHERE_SCALE*1.8f;
                if(d < cd)
                {
                    if(RENDER_PASS == 1){glUniform3f(color_id, 1.f, 0.f, 0.f);}

                    // vec sdj = spheres[j].pos;
                    // vNorm(&sdj);
                    // vReflect(&spheres[i].dir, sdj, spheres[i].dir);
                    // vNorm(&spheres[i].dir);
                    
                    // reflect the ball direction
                    vReflect(&spheres[i].dir, spheres[j].dir, spheres[i].dir);
                    vNorm(&spheres[i].dir);

                    // increment the ball to a non-inersecting distance in the new direction
                    vec inc;
                    vMulS(&inc, spheres[i].dir, (cd-d)+SPHERE_SPEED);
                    vAdd(&spheres[i].pos, spheres[i].pos, inc);

                    // char strts[16];
                    // timestamp(&strts[0]);
                    // printf("[%s] Collision %f.\n", strts, d);
                }
            }
        }
        
        if(RENDER_PASS == 1)
            rSphere(spheres[i].pos.x, spheres[i].pos.y, spheres[i].pos.z);
    }

    if(RENDER_PASS == 1)
        glfwSwapBuffers(window);
}

//*************************************
// Input Handelling
//*************************************
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // control
    if(action == GLFW_PRESS)
    {
        // new
        if(key == GLFW_KEY_N)
        {
            // end
            timeTaken(0);
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] Sim End.\n", strts);
            printf("[%s] Time-Taken: %s or %g Seconds\n\n", strts, tts, t-st);
            
            // new
            newSim();
        }

        // show average fps
        else if(key == GLFW_KEY_F)
        {
            if(t-lfct > 2.0)
            {
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] FPS: %g\n", strts, fc/(t-lfct));
                printf("[%s] LPS: %g\n", strts, lc/(t-llct));
                lfct = t;
                fc = 0;
                llct = t;
                lc = 0;
            }
        }

        // orbit in
        else if(key == GLFW_KEY_O)
        {
            for(uint i = 0; i < MAX_SPHERES; i++)
            {
                vRuvBT(&spheres[i].pos); // random point on outside of unit sphere
                vMulS(&spheres[i].pos, spheres[i].pos, (randf()*0.3f)+1.2f); // project it away from the unit sphere a little
                vRuvBT(&spheres[i].dir); // random point on outside of unit sphere
                vNorm(&spheres[i].dir); // hmm or not?
                spheres[i].c = 0.f;
            }
        }

        // toggle neural sim
        if(key == GLFW_KEY_P)
        {
            neural_sim = 1 - neural_sim;
            if(neural_sim == 1)
            {
                timeTaken(0);
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] Neural sim: ON\n", strts);
            }
            else
            {
                timeTaken(0);
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] Neural sim: OFF\n", strts);

                // reset neural colours
                for(uint i = 0; i < MAX_SPHERES; i++)
                    spheres[i].c = 0.f;
            }
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if(yoffset == -1)
        zoom += 0.06f * zoom;
    else if(yoffset == 1)
        zoom -= 0.06f * zoom;
    
    if(zoom > -2.33f){zoom = -2.33f;}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if(action == GLFW_PRESS)
    {
        if(button == GLFW_MOUSE_BUTTON_LEFT)
        {
            focus_cursor = 1 - focus_cursor;
            if(focus_cursor == 0)
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            else
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            glfwSetCursorPos(window, ww2, wh2);
            glfwGetCursorPos(window, &ww2, &wh2);
        }
        else if(button == GLFW_MOUSE_BUTTON_4 || button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if(zoom != -0.3f)
                zoom = -0.3f;
            else
                zoom = -2.33f;
        }
    }
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    winw = width;
    winh = height;

    glViewport(0, 0, winw, winh);
    aspect = (f32)winw / (f32)winh;
    ww = winw;
    wh = winh;
    rww = 1/ww;
    rwh = 1/wh;
    ww2 = ww/2;
    wh2 = wh/2;
    uw = (double)aspect / ww;
    uh = 1 / wh;
    uw2 = (double)aspect / ww2;
    uh2 = 1 / wh2;

    mIdent(&projection);
    mPerspective(&projection, 60.0f, aspect, 0.01f, FAR_DISTANCE*2.f); 
}

//*************************************
// Process Entry Point
//*************************************
int main(int argc, char** argv)
{
    // allow custom msaa level
    int msaa = 16;
    if(argc >= 2){msaa = atoi(argv[1]);}

    // allow framerate cap
    double maxfps = 144.0;
    if(argc >= 3){maxfps = atof(argv[2]);}

    // set neural_sim
    if(argc >= 4){neural_sim = atoi(argv[3]);}

    // help
    printf("----\n");
    printf("UnitCollider\n");
    printf("----\n");
    printf("James William Fletcher (github.com/mrbid)\n");
    printf("----\n");
    printf("Argv(2): msaa, maxfps\n");
    printf("e.g; ./uc 16 60\n");
    printf("----\n");

    // init glfw
    if(!glfwInit()){printf("glfwInit() failed.\n"); exit(EXIT_FAILURE);}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SAMPLES, msaa);
    window = glfwCreateWindow(winw, winh, "UnitCollider", NULL, NULL);
    if(!window)
    {
        printf("glfwCreateWindow() failed.\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    const GLFWvidmode* desktop = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (desktop->width/2)-(winw/2), (desktop->height/2)-(winh/2)); // center window on desktop
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(0); // 0 for immediate updates, 1 for updates synchronized with the vertical retrace, -1 for adaptive vsync

    // set icon
    glfwSetWindowIcon(window, 1, &(GLFWimage){16, 16, (unsigned char*)&icon_image.pixel_data});

    // hide cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);


//*************************************
// projection
//*************************************

    window_size_callback(window, winw, winh);

//*************************************
// bind vertex and index buffers
//*************************************

    // ***** BIND SPHERE *****
    esBind(GL_ARRAY_BUFFER, &mdlSphere.vid, low_vertices, sizeof(low_vertices), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlSphere.iid, low_indices, sizeof(low_indices), GL_STATIC_DRAW);

//*************************************
// compile & link shader programs
//*************************************

    //makeAllShaders();
    makeLambert();

//*************************************
// configure render options
//*************************************

    // glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    // glBlendEquation(GL_FUNC_ADD);
    
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.13, 0.13, 0.13, 0.0);

    shadeLambert(&position_id, &projection_id, &modelview_id, &lightpos_id, &color_id, &opacity_id);
    glUniform3f(lightpos_id, lightpos.x, lightpos.y, lightpos.z);
    glUniform1f(opacity_id, 1.0f);
    modelBind(&mdlSphere); // we are only using one model

//*************************************
// execute update / render loop
//*************************************

    // init
    const double maxlps = 60.0;
    t = glfwGetTime();
    lfct = t;
    dt = 1.0 / (float)maxlps; // fixed timestep delta-time
    newSim();
    
    // lps accurate event loop
    const double fps_limit = 1.0 / maxfps;
    double rlim = 0.0;
    const useconds_t wait_interval = 1000000 / maxlps; // fixed timestep
    useconds_t wait = wait_interval;
    while(!glfwWindowShouldClose(window))
    {
        usleep(wait);
        t = glfwGetTime();
        glfwPollEvents();

        if(maxfps < maxlps)
        {
            if(t > rlim)
            {
                RENDER_PASS = 1;
                rlim = t + fps_limit; // should be doing this after main_loop() at the very least but it's not a big deal.
                fc++;
            }
            else
                RENDER_PASS = 0;
        }
        else
        {
            RENDER_PASS = 1;
            fc++;
        }

        main_loop();

        // accurate lps
        wait = wait_interval - (useconds_t)((glfwGetTime() - t) * 1000000.0);
        if(wait > wait_interval)
            wait = wait_interval;
        lc++;
    }

    // end
    timeTaken(0);
    char strts[16];
    timestamp(&strts[0]);
    printf("[%s] Sim End.\n", strts);
    printf("[%s] Time-Taken: %s or %g Seconds\n\n", strts, tts, t-st);

    // done
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
    return 0;
}
