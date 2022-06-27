#define FLOAT_T float
// #define DEBUG 1

#ifndef BUFFERSIZE
#define BUFFERSIZE 1024
#endif 

#ifndef N_INPUT_RICH
#define N_INPUT_RICH 4
#endif 

#ifndef N_INPUT_MUON
#define N_INPUT_MUON 4
#endif 

#ifndef N_INPUT_GLOBALPID
#define N_INPUT_GLOBALPID 11
#endif 

#ifndef N_INPUT_GLOBALMUONID
#define N_INPUT_GLOBALMUONID 10
#endif 

#ifndef N_OUTPUT_RICH
#define N_OUTPUT_RICH 4
#endif 

#ifndef N_OUTPUT_MUON
#define N_OUTPUT_MUON 2
#endif 

#ifndef N_OUTPUT_GLOBALPID
#define N_OUTPUT_GLOBALPID 7
#endif 

#ifndef N_OUTPUT_GLOBALMUONID
#define N_OUTPUT_GLOBALMUONID 2
#endif 

#ifndef N_RANDOM_RICH
#define N_RANDOM_RICH 64
#endif 

#ifndef N_RANDOM_MUON
#define N_RANDOM_MUON 64
#endif 

#ifndef N_RANDOM_GLOBALPID
#define N_RANDOM_GLOBALPID 64
#endif 

#ifndef N_RANDOM_GLOBALMUONID
#define N_RANDOM_GLOBALMUONID 64
#endif 

#ifndef N_OUTPUT
#define N_OUTPUT 16
#endif 

#ifndef MUON_ERRORCODE
#define MUON_ERRORCODE -1000
#endif

#ifdef DEBUG
#include <stdlib.h>
#include <stdio.h>
#endif 

extern "C" FLOAT_T* RichMuon_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* RichMuon_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* RichMuon                       (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* RichPion_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* RichPion_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* RichPion                       (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* RichKaon_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* RichKaon_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* RichKaon                       (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* RichProton_tX                  (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* RichProton_tY_inverse          (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* RichProton                     (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* MuonMuon_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* MuonMuon_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* MuonMuon                       (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* MuonPion_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* MuonPion_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* MuonPion                       (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* MuonKaon_tX                    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* MuonKaon_tY_inverse            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* MuonKaon                       (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* MuonProton_tX                  (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* MuonProton_tY_inverse          (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* MuonProton                     (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalPIDMuon_tX               (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPIDMuon_tY_inverse       (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPIDMuon                  (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalPIDPion_tX               (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPIDPion_tY_inverse       (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPIDPion                  (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalPIDKaon_tX               (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPIDKaon_tY_inverse       (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPIDKaon                  (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalPIDProton_tX             (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPIDProton_tY_inverse     (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalPIDProton                (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalMuonIdMuon_tX            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonIdMuon_tY_inverse    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonIdMuon               (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalMuonIdPion_tX            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonIdPion_tY_inverse    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonIdPion               (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalMuonIdKaon_tX            (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonIdKaon_tY_inverse    (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonIdKaon               (FLOAT_T *, const FLOAT_T *);

extern "C" FLOAT_T* GlobalMuonIdProton_tX          (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonIdProton_tY_inverse  (FLOAT_T *, const FLOAT_T *);
extern "C" FLOAT_T* GlobalMuonIdProton             (FLOAT_T *, const FLOAT_T *);

typedef FLOAT_T* (*mlfun) (FLOAT_T *, const FLOAT_T *); 


extern "C"
FLOAT_T* gan_pipe ( mlfun tX, 
                    mlfun model, 
                    mlfun tY, 
                    FLOAT_T *output, 
                    const FLOAT_T *input, 
                    const FLOAT_T *random, 
                    unsigned short nIn, 
                    unsigned short nOut, 
                    unsigned short nRandom )
{
  unsigned short i; 
  FLOAT_T buf_input[BUFFERSIZE]; 
  FLOAT_T buf_output[BUFFERSIZE]; 

  tX (buf_input, input);

  #ifdef DEBUG
  printf("Preprocessed input\n");
  for (i = 0; i < nIn; ++i)
    printf("in [%d] -> pin [%d]: %.2f  ->  %.2f\n", i, i, input[i], buf_input[i]); 
  #endif

  for (i = 0; i < nRandom; ++i)
    buf_input[nIn + i] = random[i]; 

  model (buf_output, buf_input);

  tY (output, buf_output); 

#ifdef DEBUG
  printf("Preprocessed output\n");
  for (i = 0; i < nOut; ++i)
    printf("pout[%d] -> out [%d]: %.2f  ->  %.2f\n", i, i, buf_output[i], output[i]); 
#endif 

  return output; 
}


extern "C"
FLOAT_T *GenericPipe ( FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random ,
                       mlfun richTx,  mlfun richModel,  mlfun richTy  , 
                       mlfun muonTx,  mlfun muonModel,  mlfun muonTy  , 
                       mlfun gpidTx,  mlfun gpidModel,  mlfun gpidTy  , 
                       mlfun gmuidTx, mlfun gmuidModel, mlfun gmuidTy )
{
  short i, j;
  float isMuon;

  // Split the random array into four sub arrays
  j = 0; 
  const FLOAT_T* r0 = random + j;
  j += N_RANDOM_RICH; 
  const FLOAT_T* r1 = random + j;
  j += N_RANDOM_MUON; 
  const FLOAT_T* r2 = random + j;
  j += N_RANDOM_GLOBALPID; 
  const FLOAT_T* r3 = random + j;

  // Rich
  FLOAT_T richinput [N_INPUT_RICH];
  FLOAT_T richdll [N_OUTPUT_RICH];
  for (i = 0; i < N_INPUT_RICH; ++i)
    richinput[i] = input[i]; 

#ifdef DEBUG
  printf (" === RICH === \n");
  for (i = 0; i < N_INPUT_RICH; ++i)
    printf("RICH INPUT TRANSFER (%d):  %.2f  ->  %.2f\n", i, input[i], richinput[i]); 
#endif

  gan_pipe (richTx, richModel, richTy, richdll, richinput, r0, N_INPUT_RICH, N_OUTPUT_RICH, N_RANDOM_RICH); 

  // Muon
  FLOAT_T muoninput [N_INPUT_MUON];
  FLOAT_T muondll [N_OUTPUT_MUON];
  for (i = 0; i < N_INPUT_MUON; ++i)
    muoninput[i] = input[i]; 

#ifdef DEBUG
  printf (" === MUON === \n");
#endif
  isMuon = input[N_INPUT_RICH];

  if (isMuon > 0.5){
    gan_pipe (muonTx, muonModel, muonTy, muondll, muoninput, r1, N_INPUT_RICH, N_OUTPUT_RICH, N_RANDOM_RICH);
    muondll[1] = muondll[0] - muondll[1];   // current model returns muon LL diff
  }
  else
    for (i = 0; i < N_OUTPUT_MUON; ++i)
      muondll[i] = MUON_ERRORCODE; 

  // Global PID
  FLOAT_T gpid_input [N_INPUT_GLOBALPID];
  FLOAT_T gpid_output [N_OUTPUT_GLOBALPID];
  j = 0;

  // p, eta, nTracks, charge
  for (i = 0; i < N_INPUT_RICH; ++i)
    gpid_input[j++] = input[i]; 
  
  // dlle, dllmu, dllk, dllp
  for (i = 0; i < N_OUTPUT_RICH; ++i)
    gpid_input[j++] = richdll[i]; 

  // isMuon 
  gpid_input[j++] = isMuon; 

  // mullmu, mullbg
  for (i = 0; i < N_OUTPUT_MUON; ++i)
    gpid_input[j++] = muondll[i];

#ifdef DEBUG
  printf (" === GLOBAL PID === \n");
#endif
  gan_pipe (gpidTx, gpidModel, gpidTy, gpid_output, gpid_input, r2, N_INPUT_GLOBALPID, N_OUTPUT_GLOBALPID, N_RANDOM_GLOBALPID); 
 
  // Global Muon ID
  FLOAT_T gmuid_input [N_INPUT_GLOBALMUONID];
  FLOAT_T gmuid_output [N_OUTPUT_GLOBALMUONID];
  j = 0;

  // p, eta, nTracks, charge
  for (i = 0; i < N_INPUT_RICH; ++i)
    gmuid_input[j++] = input[i]; 
  
  // dlle, dllmu, dllk, dllp
  for (i = 0; i < N_OUTPUT_RICH; ++i)
    gmuid_input[j++] = richdll[i]; 

  // mullmu, mullbg
  for (i = 0; i < N_OUTPUT_MUON; ++i)
    gmuid_input[j++] = muondll[i]; 

#ifdef DEBUG
  printf (" === GLOBAL MUON ID === \n");
#endif
  gan_pipe (gmuidTx, gmuidModel, gmuidTy, gmuid_output, gmuid_input, r3, N_INPUT_GLOBALMUONID, N_OUTPUT_GLOBALMUONID, N_RANDOM_GLOBALMUONID); 
  
  // Format output 
  j = 0;
  for (i = 0; i < N_OUTPUT_RICH; ++i)
    output[j++] = richdll[i]; 

  for (i = 0; i < N_OUTPUT_MUON; ++i)
    output[j++] = muondll[i]; 
  
  for (i = 0; i < N_OUTPUT_GLOBALPID; ++i)
    output[j++] = gpid_output[i]; 

  for (i = 0; i < N_OUTPUT_GLOBALMUONID; ++i)
    output[j++] = gmuid_output[i]; 

  return output; 
}


extern "C"
FLOAT_T *muon_pipe (FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random)
{
#ifdef DEBUG
  int i = 0;
  printf ("Pion pipe: INPUT\n") ;
  for (i = 0; i < N_INPUT_RICH; ++i)
    printf ( "in [%d]: %.2f\n", i, input[i]);

  printf ("Pion pipe: RANDOM\n") ;
  for (i = 0; i < N_RANDOM_RICH; ++i)
    printf ( "rnd [%d]: %.2f\n", i, random[i]);
#endif 

  return GenericPipe ( output , input , random ,
                       RichMuon_tX         , RichMuon         , RichMuon_tY_inverse         ,
                       MuonMuon_tX         , MuonMuon         , MuonMuon_tY_inverse         ,
                       GlobalPIDMuon_tX    , GlobalPIDMuon    , GlobalPIDMuon_tY_inverse    ,
                       GlobalMuonIdMuon_tX , GlobalMuonIdMuon , GlobalMuonIdMuon_tY_inverse ); 
}


extern "C"
FLOAT_T *pion_pipe (FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random)
{
#ifdef DEBUG
  int i = 0;
  printf ("Pion pipe: INPUT\n") ;
  for (i = 0; i < N_INPUT_RICH; ++i)
    printf ( "in [%d]: %.2f\n", i, input[i]);

  printf ("Pion pipe: RANDOM\n") ;
  for (i = 0; i < N_RANDOM_RICH; ++i)
    printf ( "rnd [%d]: %.2f\n", i, random[i]);
#endif 

  return GenericPipe ( output , input , random ,
                       RichPion_tX         , RichPion         , RichPion_tY_inverse         ,
                       MuonPion_tX         , MuonPion         , MuonPion_tY_inverse         ,
                       GlobalPIDPion_tX    , GlobalPIDPion    , GlobalPIDPion_tY_inverse    ,
                       GlobalMuonIdPion_tX , GlobalMuonIdPion , GlobalMuonIdPion_tY_inverse ); 
}


extern "C"
FLOAT_T *kaon_pipe (FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random)
{
#ifdef DEBUG
  int i = 0;
  printf ("Pion pipe: INPUT\n") ;
  for (i = 0; i < N_INPUT_RICH; ++i)
    printf ( "in [%d]: %.2f\n", i, input[i]);

  printf ("Pion pipe: RANDOM\n") ;
  for (i = 0; i < N_RANDOM_RICH; ++i)
    printf ( "rnd [%d]: %.2f\n", i, random[i]);
#endif 

  return GenericPipe ( output , input , random ,
      RichKaon_tX         , RichKaon         , RichKaon_tY_inverse         ,
      MuonKaon_tX         , MuonKaon         , MuonKaon_tY_inverse         ,
      GlobalPIDKaon_tX    , GlobalPIDKaon    , GlobalPIDKaon_tY_inverse    ,
      GlobalMuonIdKaon_tX , GlobalMuonIdKaon , GlobalMuonIdKaon_tY_inverse ); 
}


extern "C"
FLOAT_T *proton_pipe (FLOAT_T* output, const FLOAT_T *input, const FLOAT_T *random)
{
#ifdef DEBUG
  int i = 0;
  printf ("Pion pipe: INPUT\n") ;
  for (i = 0; i < N_INPUT_RICH; ++i)
    printf ( "in [%d]: %.2f\n", i, input[i]);

  printf ("Pion pipe: RANDOM\n") ;
  for (i = 0; i < N_RANDOM_RICH; ++i)
    printf ( "rnd [%d]: %.2f\n", i, random[i]);
#endif 

  return GenericPipe ( output , input , random ,
      RichProton_tX         , RichProton         , RichProton_tY_inverse         ,
      MuonProton_tX         , MuonProton         , MuonProton_tY_inverse         ,
      GlobalPIDProton_tX    , GlobalPIDProton    , GlobalPIDProton_tY_inverse    ,
      GlobalMuonIdProton_tX , GlobalMuonIdProton , GlobalMuonIdProton_tY_inverse ); 
}
