/*
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#ifndef _NVGRAPH_H_
#define _NVGRAPH_H_

#include "stddef.h"
#include "cuda_runtime_api.h"
#include <library_types.h>

#ifndef NVGRAPH_API
#ifdef _WIN32
#define NVGRAPH_API __stdcall
#else
#define NVGRAPH_API
#endif
#endif

#ifdef __cplusplus
  extern "C" {
#endif

/* nvGRAPH status type returns */
typedef enum
{
    NVGRAPH_STATUS_SUCCESS            =0,
    NVGRAPH_STATUS_NOT_INITIALIZED    =1,
    NVGRAPH_STATUS_ALLOC_FAILED       =2,
    NVGRAPH_STATUS_INVALID_VALUE      =3,
    NVGRAPH_STATUS_ARCH_MISMATCH      =4,
    NVGRAPH_STATUS_MAPPING_ERROR      =5,
    NVGRAPH_STATUS_EXECUTION_FAILED   =6,
    NVGRAPH_STATUS_INTERNAL_ERROR     =7,
    NVGRAPH_STATUS_TYPE_NOT_SUPPORTED =8,
    NVGRAPH_STATUS_NOT_CONVERGED      =9

} nvgraphStatus_t;

const char* nvgraphStatusGetString( nvgraphStatus_t status);

/* Opaque structure holding nvGRAPH library context */
struct nvgraphContext;
typedef struct nvgraphContext *nvgraphHandle_t;

/* Opaque structure holding the graph descriptor */
struct nvgraphGraphDescr;
typedef struct nvgraphGraphDescr *nvgraphGraphDescr_t;

/* Semi-ring types */
typedef enum
{
   NVGRAPH_PLUS_TIMES_SR = 0,
   NVGRAPH_MIN_PLUS_SR   = 1,
   NVGRAPH_MAX_MIN_SR    = 2,
   NVGRAPH_OR_AND_SR     = 3,
} nvgraphSemiring_t;

/* Topology types */
typedef enum
{
   NVGRAPH_CSR_32 = 0,
   NVGRAPH_CSC_32 = 1,
   NVGRAPH_COO_32 = 2,
} nvgraphTopologyType_t;

typedef enum
{
   NVGRAPH_DEFAULT                = 0,  // Default is unsorted.
   NVGRAPH_UNSORTED               = 1,  //
   NVGRAPH_SORTED_BY_SOURCE       = 2,  // CSR
   NVGRAPH_SORTED_BY_DESTINATION  = 3   // CSC
} nvgraphTag_t;

struct nvgraphCSRTopology32I_st {
  int nvertices; // n+1
  int nedges; // nnz
  int *source_offsets; // rowPtr
  int *destination_indices; // colInd
};
typedef struct nvgraphCSRTopology32I_st *nvgraphCSRTopology32I_t;

struct nvgraphCSCTopology32I_st {
  int nvertices; // n+1
  int nedges; // nnz
  int *destination_offsets; // colPtr
  int *source_indices; // rowInd
};
typedef struct nvgraphCSCTopology32I_st *nvgraphCSCTopology32I_t;

struct nvgraphCOOTopology32I_st {
  int nvertices; // n+1
  int nedges; // nnz
  int *source_indices; // rowInd
  int *destination_indices; // colInd
  nvgraphTag_t tag;
};
typedef struct nvgraphCOOTopology32I_st *nvgraphCOOTopology32I_t;

/* Open the library and create the handle */
nvgraphStatus_t NVGRAPH_API nvgraphCreate(nvgraphHandle_t *handle);

/*  Close the library and destroy the handle  */
nvgraphStatus_t NVGRAPH_API nvgraphDestroy(nvgraphHandle_t handle);

/* Create an empty graph descriptor */
nvgraphStatus_t NVGRAPH_API nvgraphCreateGraphDescr(nvgraphHandle_t handle, nvgraphGraphDescr_t *descrG);

/* Destroy a graph descriptor */
nvgraphStatus_t NVGRAPH_API nvgraphDestroyGraphDescr(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG);

/* Set size, topology data in the graph descriptor  */
nvgraphStatus_t NVGRAPH_API nvgraphSetGraphStructure(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void* topologyData, nvgraphTopologyType_t TType);

/* Query size and topology information from the graph descriptor */
nvgraphStatus_t NVGRAPH_API nvgraphGetGraphStructure (nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void* topologyData, nvgraphTopologyType_t* TType);

/* Allocate numsets vectors of size V reprensenting Vertex Data and attached them the graph.
 * settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type */
nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, size_t numsets, cudaDataType_t  *settypes);

/* Allocate numsets vectors of size E reprensenting Edge Data and attached them the graph.
 * settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type */
nvgraphStatus_t NVGRAPH_API nvgraphAllocateEdgeData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, size_t numsets, cudaDataType_t *settypes);

/* Update the vertex set #setnum with the data in *vertexData, sets have 0-based index
 *  Conversions are not sopported so nvgraphTopologyType_t should match the graph structure */
nvgraphStatus_t NVGRAPH_API nvgraphSetVertexData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void *vertexData, size_t setnum);

/* Copy the edge set #setnum in *edgeData, sets have 0-based index
 *  Conversions are not sopported so nvgraphTopologyType_t should match the graph structure */
nvgraphStatus_t NVGRAPH_API nvgraphGetVertexData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void *vertexData, size_t setnum);

/* Convert the edge data to another topology
 */
nvgraphStatus_t NVGRAPH_API nvgraphConvertTopology(nvgraphHandle_t handle,
                                nvgraphTopologyType_t srcTType, void *srcTopology, void *srcEdgeData, cudaDataType_t *dataType,
                                nvgraphTopologyType_t dstTType, void *dstTopology, void *dstEdgeData);

/* Convert graph to another structure
 */
nvgraphStatus_t NVGRAPH_API nvgraphConvertGraph(nvgraphHandle_t handle, nvgraphGraphDescr_t srcDescrG, nvgraphGraphDescr_t dstDescrG, nvgraphTopologyType_t dstTType);

/* Update the edge set #setnum with the data in *edgeData, sets have 0-based index
 *  Conversions are not sopported so nvgraphTopologyType_t should match the graph structure */
nvgraphStatus_t NVGRAPH_API nvgraphSetEdgeData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void *edgeData, size_t setnum);

/* Copy the edge set #setnum in *edgeData, sets have 0-based index
 * Conversions are not sopported so nvgraphTopologyType_t should match the graph structure */
nvgraphStatus_t NVGRAPH_API nvgraphGetEdgeData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void *edgeData, size_t setnum);

/* create a new graph by extracting a subgraph given a list of vertices
 */
nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByVertex(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, nvgraphGraphDescr_t subdescrG, int *subvertices, size_t numvertices );
/* create a new graph by extracting a subgraph given a list of edges
 */
nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByEdge( nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, nvgraphGraphDescr_t subdescrG, int *subedges , size_t numedges);

/* nvGRAPH Semi-ring sparse matrix vector multiplication
 */
nvgraphStatus_t NVGRAPH_API nvgraphSrSpmv(nvgraphHandle_t handle,
                                 const nvgraphGraphDescr_t descrG,
                                 const size_t weight_index,
                                 const void *alpha,
                                 const size_t x_index,
                                 const void *beta,
                                 const size_t y_index,
                                 const nvgraphSemiring_t SR);

/* nvGRAPH Single Source Shortest Path (SSSP)
 * Calculate the shortest path distance from a single vertex in the graph to all other vertices.
 */
nvgraphStatus_t NVGRAPH_API nvgraphSssp(nvgraphHandle_t handle,
                               const nvgraphGraphDescr_t descrG,
                               const size_t weight_index,
                               const int *source_vert,
                               const size_t sssp_index);

/* nvGRAPH WidestPath
 * Find widest path potential from source_index to every other vertices.
 */
nvgraphStatus_t NVGRAPH_API nvgraphWidestPath(nvgraphHandle_t handle,
                                  const nvgraphGraphDescr_t descrG,
                                  const size_t weight_index,
                                  const int *source_vert,
                                  const size_t widest_path_index);

/* nvGRAPH PageRank
 * Find PageRank for each vertex of a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
 */
nvgraphStatus_t NVGRAPH_API nvgraphPagerank(nvgraphHandle_t handle,
                                   const nvgraphGraphDescr_t descrG,
                                   const size_t weight_index,
                                   const void *alpha,
                                   const size_t bookmark_index,
                                   const int has_guess,
                                   const size_t pagerank_index,
                                   const float tolerance,
                                   const int max_iter );

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NVGRAPH_H_ */

