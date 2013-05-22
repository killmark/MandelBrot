#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/timeb.h>
#include <time.h>
#include "mpi.h"

/* two strategy */
void dynamic_strategy(int);
void static_strategy(int, int);

/* check wether c = a + bi is in the MandelBrot set */
int isMandelBrot(double, double);
/* for static strategy */
void worker(int, int);
/* for dynamic strategy */
void master();
void slave();

/* global variable */
const int work_tag = 1;
const int die_tag = 0;
const int image_size = 10000;
const int max_repeats = 255;
int chunk_size;
int width;
int height;
double accu;
char* result = NULL;
FILE* fp = NULL;

long long getTime() {
    struct timeb t;
    ftime(&t);
    return 1000 * t.time + t.millitm;
}

int main(int argc, char* argv[]){
    int nprocs;                 
    int procid;
    int stat;
    int i, j;
    int opt;
    
    /* get strategy and chunk size */
    opt = atoi(argv[1]);
    
    /* do this first to init MPI */
    MPI_Init(&argc, &argv);  

    /* return number of procs */
    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert (stat == MPI_SUCCESS);

    /* my integer proc id */
    stat = MPI_Comm_rank(MPI_COMM_WORLD, &procid); 
    assert (stat == MPI_SUCCESS);
    
    /* do check first */
    /* 1: static strategy 
       2: dynamic strategy */
    if(procid == 0){
        if(opt != 1 && opt != 2){
            printf("Error: strategy number should either 1 or 2\n");
	    return -1;
        }
	if(opt == 1 && argc < 2){
            printf("USAGE: ./hw1 1");
	    return -1;
	}
        if(opt == 2 && argc < 3){
            printf("USAGE: ./hw1 2 <chunk size>\n");
            return -1;
        }

    }
    
    /* set width and height */
    width = image_size;
    height = image_size;
    accu = image_size/4.0;

    /* choosing stategy */
    if(opt == 1)
        static_strategy(nprocs, procid);
    else{
        chunk_size = atoi(argv[2]);
        dynamic_strategy(procid);
    }
    
    if(opt == 2){
        /* output matrix as a file */
        if(procid == 0){
            fp = fopen("image", "w");
            for(i = 0; i < image_size*image_size; ++i){
                fprintf(fp, "%c ", result[i]);
                if((i + 1)%image_size == 0)
                    fputs("\n", fp);
            }
        }
    
        /* terminated cleanly */
        if(result != NULL)
            free(result);
        if(fp != NULL)
            fclose(fp);
    }
    
    MPI_Finalize();
    return 0;
}
void master(){

    int i;
    int procid;
    int nprocs, slave_num;
    int stat;
    MPI_Status status;
    int arr_size;
    int send_offset = 0;
    char* temp;
    clock_t start, end;

    /* master just schedule the slaves */
    /* master will not do any calculation */

    /* get the total number of processors */
    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert (stat == MPI_SUCCESS);

    /* init */
    slave_num = nprocs - 1;
    arr_size = width*height;
    
    result = (char*) malloc (arr_size*(sizeof(char)));
    
    for(i = 0; i < arr_size; ++i){
        result[i] = '0';
    }

    temp = (char*) malloc (chunk_size*sizeof(char));
    
    start = clock();
    for(procid = 1; procid <= slave_num; ++procid){
        if(send_offset != arr_size){
            MPI_Send(&send_offset, 1, MPI_INT, procid, work_tag, MPI_COMM_WORLD);
            send_offset += chunk_size;
        }
        else{
            slave_num = procid - 1;
            break;
        }
    }
    while(send_offset != arr_size){
        /* receive the data to temp */
        MPI_Recv(temp, chunk_size, MPI_CHAR, MPI_ANY_SOURCE,
                 MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        /* copy temp to the final position */
	/* position is determined by MPI_TAG received */
        strncpy(result + status.MPI_TAG, temp, chunk_size);
        /* send offset again */
        MPI_Send(&send_offset, 1, MPI_INT, status.MPI_SOURCE, work_tag, MPI_COMM_WORLD);
        send_offset += chunk_size;
    }
    
    for(procid = 1; procid <= slave_num; ++procid){
        MPI_Recv(temp, chunk_size, MPI_CHAR, procid,
                 MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        strncpy(result + status.MPI_TAG, temp, chunk_size);
    }
    /* Tell all the slaves to exit by sending an empty message with the
       DIETAG. */
    for(procid = 1; procid < nprocs; ++procid){
        MPI_Send(0, 0, MPI_INT, procid, die_tag, MPI_COMM_WORLD);
    }
    free(temp);
    end = clock();
    printf("Time: %f s.\n", ((double)(end - start))/CLOCKS_PER_SEC);

}
void slave(){
    char* mat;
    int i, j;
    int x, y;
    MPI_Status status;
    int offset;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    
    mat = (char*) malloc (chunk_size*sizeof(char));
    for(j = 0; j < chunk_size; j++){
        mat[j] = '1';
    }
    while(1){
        MPI_Recv(&offset, 1, MPI_INT, 0,
                 MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(status.MPI_TAG == die_tag){
	    /* if receive die_tag, exit */
            break;
        }
        for(i = offset; i < chunk_size + offset; ++i){
            x = i%width;
            y = i/width;
	    /* just set the accuracy 1/accu (distance between 2 points) */
	    double a = (double) (x - width/2)/accu;
            double b = (double) (height/2 - y)/accu;
            if(isMandelBrot(a, b)){
                mat[i - offset] = '0';
            }
        }
        MPI_Send(mat, chunk_size, MPI_CHAR, 0, offset, MPI_COMM_WORLD);
	/* reset mat */
        for(j = 0; j < chunk_size; j++){
            mat[j] = '1';
        }
    }
    free(mat);
}

void worker(int nprocs, int procid){
    int lines = height/nprocs;
    int begin, end;
    char* mat;
    int i, j;
    int x, y;
    int mat_size;
    FILE* fp = NULL;
    clock_t start, endd;
    double duration;
    double statMax = 0.0;

    start = clock();
    if(procid != nprocs - 1){
        begin = procid*lines;
        end = begin + lines;
        mat = (char*) malloc (lines*width*sizeof(int));
        mat_size = lines*width;
    }
    else{
        /* last one maybe different */
        begin = procid*lines;
        end = width;
        mat = (char*) malloc ((height*width - procid*lines*width)*sizeof(int));
        mat_size = height*width - procid*lines*width;
    }
    /* init mat */
    for(i = 0; i < mat_size; ++i){
        mat[i] = '1';
    }
    for(y = begin; y < end; ++y){
        for(x = 0; x < width; ++x){
            double a = (double) (x - width/2)/accu;
            double b = (double) (height/2 - y)/accu;
            if(isMandelBrot(a, b)){
                mat[(y-begin)*width + x] = '0';
            }
        }
    }
    endd = clock();
    duration = ((double) (endd - start))/CLOCKS_PER_SEC;
    /* get max time */
    MPI_Reduce(&duration,&statMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(procid == 0)
      printf("Time: %f s.\n", statMax);

    /* create image */
    /* write file one by one */
    for (i = 0; i < nprocs; ++i){
        if (i == procid){
            fp = fopen("image", "a");
            for(j = 0; j < mat_size; ++j){
                fprintf(fp, "%c ", mat[j]);
                if((j + 1)%width == 0)
                    fputs("\n", fp);
            }
            fclose(fp);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    free(mat);
}

void static_strategy(int nprocs, int procid){
    worker(nprocs, procid);
}

void dynamic_strategy(int procid){
    if(procid == 0){
        master();
    }
    else{
        slave();
    }
}


int isMandelBrot(double a, double b){
    int repeats = 0;
    double x = 0.0;
    double y = 0.0;
    do{
        double temp = x;
        x = x*x - y*y + a;
        y = 2*temp*y + b;
        ++repeats;
    }while(x*x + y*y <= 4.0 && repeats < max_repeats);
    
    /* check whether c = a + bi is in the MandelBrot set */
    if(repeats == max_repeats)
        return 1;
    else
        return 0;
}
