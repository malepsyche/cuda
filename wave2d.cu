#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

#define TILE 16

#define CUDA_CHECK(x) do { cudaError_t err = x; if(err != cudaSuccess){ \
std::cerr<<"CUDA error: "<<cudaGetErrorString(err)<<std::endl; exit(1);} } while(0)

// ================= TIMER =================
struct GPUTimer {
    cudaEvent_t start, stop;
    GPUTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    void tic() { cudaEventRecord(start); }
    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ================= GLOBAL =================
__global__ void wave_global(double* next, double* curr, double* prev,
                           int Nx, int Ny, double lambda2) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i<=0 || j<=0 || i>=Ny-1 || j>=Nx-1) return;

    int id = i*Nx + j;

    double lap =
        curr[id+1] + curr[id-1] +
        curr[id+Nx] + curr[id-Nx] -
        4.0*curr[id];

    next[id] = 2.0*curr[id] - prev[id] + lambda2*lap;
}

// ================= SHARED =================
__global__ void wave_shared(double* next, double* curr, double* prev,
                           int Nx, int Ny, double lambda2) {

    __shared__ double tile[TILE+2][TILE+2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = blockIdx.x*TILE + tx;
    int i = blockIdx.y*TILE + ty;

    int lx = tx+1, ly = ty+1;

    if(i<Ny && j<Nx)
        tile[ly][lx] = curr[i*Nx+j];

    if(tx==0 && j>0) tile[ly][0] = curr[i*Nx+j-1];
    if(tx==TILE-1 && j<Nx-1) tile[ly][TILE+1] = curr[i*Nx+j+1];

    if(ty==0 && i>0) tile[0][lx] = curr[(i-1)*Nx+j];
    if(ty==TILE-1 && i<Ny-1) tile[TILE+1][lx] = curr[(i+1)*Nx+j];

    __syncthreads();

    if (i<=0 || j<=0 || i>=Ny-1 || j>=Nx-1) return;

    double lap =
        tile[ly][lx+1] + tile[ly][lx-1] +
        tile[ly+1][lx] + tile[ly-1][lx] -
        4.0*tile[ly][lx];

    int id = i*Nx+j;
    next[id] = 2.0*curr[id] - prev[id] + lambda2*lap;
}

// ================= INIT =================
void init(std::vector<double>& u, int Nx, int Ny) {
    for(int i=0;i<Ny;i++)
        for(int j=0;j<Nx;j++){
            double x = j*1.0/(Nx-1);
            double y = i*1.0/(Ny-1);
            u[i*Nx+j] = sin(M_PI*x)*sin(M_PI*y);
        }
}

// ================= SAVE =================
void save(double* d_u, int Nx, int Ny, int step) {
    std::vector<double> h(Nx*Ny);
    cudaMemcpy(h.data(), d_u, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);

    std::ofstream f("frame_"+std::to_string(step)+".txt");
    for(int i=0;i<Ny;i++){
        for(int j=0;j<Nx;j++)
            f << h[i*Nx+j] << " ";
        f << "\n";
    }
}

// ================= cuSPARSE =================
void buildCSR(int N, int Nx,
              std::vector<int>& rowPtr,
              std::vector<int>& colInd,
              std::vector<double>& val) {

    int nnz=0;
    rowPtr.resize(N+1);

    for(int i=0;i<N;i++){
        rowPtr[i]=nnz;

        int x=i%Nx, y=i/Nx;

        colInd.push_back(i); val.push_back(-4); nnz++;

        if(x>0){ colInd.push_back(i-1); val.push_back(1); nnz++; }
        if(x<Nx-1){ colInd.push_back(i+1); val.push_back(1); nnz++; }
        if(y>0){ colInd.push_back(i-Nx); val.push_back(1); nnz++; }
        if(y<Nx-1){ colInd.push_back(i+Nx); val.push_back(1); nnz++; }
    }
    rowPtr[N]=nnz;
}

// ================= MAIN =================
int main(int argc, char** argv){

    int baseN = 256;
    int L = (argc>1)?atoi(argv[1]):1;

    int Nx = baseN * L;
    int Ny = baseN * L;
    int steps = 500;

    double c=1.0, dx=0.01, dt=0.005;
    double lambda = c*dt/dx;
    double lambda2 = lambda*lambda;

    size_t size = Nx*Ny*sizeof(double);

    std::vector<double> h_prev(Nx*Ny), h_curr(Nx*Ny);
    init(h_curr,Nx,Ny);
    h_prev = h_curr;

    double *d_prev,*d_curr,*d_next;
    cudaMalloc(&d_prev,size);
    cudaMalloc(&d_curr,size);
    cudaMalloc(&d_next,size);

    cudaMemcpy(d_prev,h_prev.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr,h_curr.data(),size,cudaMemcpyHostToDevice);

    dim3 block(TILE,TILE);
    dim3 grid((Nx+TILE-1)/TILE,(Ny+TILE-1)/TILE);

    GPUTimer timer;
    timer.tic();

    for(int t=0;t<steps;t++){
        wave_shared<<<grid,block>>>(d_next,d_curr,d_prev,Nx,Ny,lambda2);

        std::swap(d_prev,d_curr);
        std::swap(d_curr,d_next);

        if(t%100==0) save(d_curr,Nx,Ny,t);
    }

    float ms = timer.toc();

    double bytes = Nx*Ny*48.0*steps;
    double bandwidth = bytes / (ms/1000.0) / 1e9;

    std::cout<<"L="<<L<<" Nx="<<Nx<<"\n";
    std::cout<<"Time(ms): "<<ms<<"\n";
    std::cout<<"Bandwidth(GB/s): "<<bandwidth<<"\n";

    std::ofstream log("results.csv", std::ios::app);
    log<<L<<","<<Nx<<","<<ms<<","<<bandwidth<<"\n";

    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_next);

    return 0;
}