# def align_dtw(alignment_index1, alignment_index2 , queryTrace):
# #     ```
# #     alignmentObj: alignment object from dtw-python call
# #     queryTrace: the query (warped) variable from the dtw-python call
# #     ```
#     import numpy as np
#     idx_multi_bool = np.diff(alignment_index2)==0
#     idx_multi_bool = np.hstack([idx_multi_bool , 0]) + np.hstack([0 , idx_multi_bool])
#     idx_multi = alignment_index2[np.where(idx_multi_bool)[0]]

#     output = np.zeros(len(alignment_index2)) * np.nan
#     for ii in range(len(alignment_index2)):
#         if idx_multi_bool[ii]:
#             tmp_idx_multi = alignment_index2==alignment_index2[ii]
#             output[alignment_index2[ii]] = np.mean(queryTrace[alignment_index1[tmp_idx_multi]])
#         else:
#             output[alignment_index2[ii]] = queryTrace[alignment_index1[ii]]
#     output = output[np.isnan(output)==0]
#     return output

# import dtw
# from tqdm import trange
# import numpy as np

# ## traces need to be pretty similar to use this code

# ## this dtw library is poorly documented and kind of janky. It won't let me put in large sequences. Anything over
# # 40k samples takes up 100s of GB of memory. f that. so I wrote a thing to split it up into chunks. There are some
# # edge effects, so the chunking also has some overlaps that are removed.

# ## The main outputs here are the matched indices: alignment_index1 and alignment_index2. Use these with the align_dtw
# # function to warp a signal

# input_template = scipy.signal.savgol_filter(np.diff(X_cc[:,0]) , 31,3)
# input_query = -scipy.signal.savgol_filter(np.diff(X_cc[:,1]) , 31,3)

# last_good_idx_query = int(np.max(np.where(np.isnan(input_query)==0)[0]))

# template = input_template[:last_good_idx_query]
# query = input_query[:last_good_idx_query]

# chunk_size = 10000
# overlap_size = 1000 # make even and the length of the window of possible shifts

# num_chunks = int(np.ceil(len(template)/chunk_size))

# alignment_list_extended = list(np.ones(num_chunks))
# alignment_list_index1 = list(np.ones(num_chunks))
# alignment_list_index2 = list(np.ones(num_chunks))
# for ii in trange(num_chunks):
#     idx_chunk_extended = np.arange(np.maximum(ii*chunk_size - overlap_size , 0), 
#                                    np.minimum( (ii+1)*chunk_size + overlap_size , len(template)))
#     template_chunk = template[idx_chunk_extended]
#     query_chunk = query[idx_chunk_extended]
#     alignment_list_extended[ii] = dtw.dtw(query_chunk , template_chunk, 
#                     keep_internals=True,
#                    step_pattern=dtw.rabinerJuangStepPattern(6,"c"),
# #                     window_type="sakoechiba", window_args={'window_size':10000}
#                    )
    
#     if ii==0:
#         idx_chunk = np.arange(0, np.min(np.where(alignment_list_extended[ii].index2==chunk_size)[0]))
#     elif ii==num_chunks-1:
#         idx_chunk = np.arange( np.min(np.where(alignment_list_extended[ii].index2==overlap_size)[0]),
#                       len(alignment_list_extended[ii].index2))
#     else:
#         idx_chunk = np.arange( np.min(np.where(alignment_list_extended[ii].index2==overlap_size)[0]),
#                               np.min(np.where(alignment_list_extended[ii].index2==chunk_size+overlap_size)[0]))
#     alignment_list_index1[ii] = alignment_list_extended[ii].index1[idx_chunk]
#     alignment_list_index2[ii] = alignment_list_extended[ii].index2[idx_chunk]

# alignment_index1 = []
# alignment_index2 = []
# for ii in range(num_chunks):
#     if ii==0:
#         alignment_index1 = np.uint64(np.hstack(( alignment_index1 ,
#                                             (alignment_list_index1[ii] + ii*chunk_size)) ))
#         alignment_index2 = np.uint64(np.hstack(( alignment_index2 ,
#                                             (alignment_list_index2[ii] + ii*chunk_size)) ))
#     else:
#         alignment_index1 = np.uint64(np.hstack(( alignment_index1 ,
#                                                 (alignment_list_index1[ii] + ii*chunk_size - overlap_size)) ))
#         alignment_index2 = np.uint64(np.hstack(( alignment_index2 ,
#                                                 (alignment_list_index2[ii] + ii*chunk_size - overlap_size)) ))


# ######
# np.corrcoef(template , output)[0,1]**2
# ######
# output = align_dtw(alignment_index1, alignment_index2 , query)

# plt.figure()
# plt.plot(alignment_index1)
# plt.plot(alignment_index2)

# plt.figure()
# # plt.plot(scipy.signal.savgol_filter(np.diff(alignment_index1) , 301,3))
# # plt.plot(scipy.signal.savgol_filter(np.diff(alignment_index2) , 301,3))
# # plt.plot(np.array(alignment_index1 , dtype='float64') - np.array(alignment_index2 , dtype='float64'))
# plt.plot(alignment_index1 , np.array(alignment_index1 , dtype='float64') - np.array(alignment_index2 , dtype='float64'))

# plt.figure()
# plt.plot(template)
# plt.plot(query)
# plt.plot(output)
# # plt.plot(scipy.signal.savgol_filter(np.diff(alignment_index1) , 301,3) - scipy.signal.savgol_filter(np.diff(alignment_index2) , 301,3))
# plt.plot(alignment_index1 , (np.array(alignment_index1 , dtype='float64') - np.array(alignment_index2 , dtype='float64')) / 30)
# plt.plot(X_cc[:,0]/10)


# # decoder_output_S2pWarped = align_dtw(alignment_index1, alignment_index2 , decoder_output)
