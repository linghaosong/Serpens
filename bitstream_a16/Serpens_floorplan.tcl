puts "applying partitioning constraints generated by tapac"
write_checkpoint before_add_floorplan_constraints.dcp

# begin defining a slot
create_pblock CR_X4Y0_To_CR_X7Y3
resize_pblock CR_X4Y0_To_CR_X7Y3 -add CLOCKREGION_X4Y0:CLOCKREGION_X7Y3
# remove the reserved clock regions for the Vitis infra
resize_pblock CR_X4Y0_To_CR_X7Y3 -remove CLOCKREGION_X7Y1:CLOCKREGION_X7Y11


# begin defining a slot
create_pblock CR_X0Y4_To_CR_X3Y7
resize_pblock CR_X0Y4_To_CR_X3Y7 -add CLOCKREGION_X0Y4:CLOCKREGION_X3Y7
# remove the reserved clock regions for the Vitis infra
resize_pblock CR_X0Y4_To_CR_X3Y7 -remove CLOCKREGION_X7Y1:CLOCKREGION_X7Y11


# begin defining a slot
create_pblock CR_X0Y8_To_CR_X3Y11
resize_pblock CR_X0Y8_To_CR_X3Y11 -add CLOCKREGION_X0Y8:CLOCKREGION_X3Y11
# remove the reserved clock regions for the Vitis infra
resize_pblock CR_X0Y8_To_CR_X3Y11 -remove CLOCKREGION_X7Y1:CLOCKREGION_X7Y11


# begin defining a slot
create_pblock CR_X4Y4_To_CR_X7Y7
resize_pblock CR_X4Y4_To_CR_X7Y7 -add CLOCKREGION_X4Y4:CLOCKREGION_X7Y7
# remove the reserved clock regions for the Vitis infra
resize_pblock CR_X4Y4_To_CR_X7Y7 -remove CLOCKREGION_X7Y1:CLOCKREGION_X7Y11


# begin defining a slot
create_pblock CR_X0Y0_To_CR_X3Y3
resize_pblock CR_X0Y0_To_CR_X3Y3 -add CLOCKREGION_X0Y0:CLOCKREGION_X3Y3
# remove the reserved clock regions for the Vitis infra
resize_pblock CR_X0Y0_To_CR_X3Y3 -remove CLOCKREGION_X7Y1:CLOCKREGION_X7Y11

add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -regex {
  pfm_top_i/dynamic_region/.*/inst/.*/control_s_axi_U_slr_0
  pfm_top_i/dynamic_region/.*/inst/.*/tapa_state.*
} ]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR1] [get_cells -regex {
  pfm_top_i/dynamic_region/.*/inst/.*/control_s_axi_U_slr_1
} ]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR2] [get_cells -regex {
  pfm_top_i/dynamic_region/.*/inst/.*/control_s_axi_U_slr_2
} ]
add_cells_to_pblock [get_pblocks CR_X4Y0_To_CR_X7Y3] [get_cells -regex {
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_0
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_1
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_1__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_0
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_1
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_1__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_2
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_2__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_3
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_3__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_0
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_1
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_1__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_2
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_2__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_3
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_3__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_0
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_1
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_1__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_2
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_2__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_3
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_3__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_13
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_13__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_14
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_14__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_15
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_15__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_X_0
  pfm_top_i/dynamic_region/.*/inst/.*/read_X_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_edge_list_ptr_0
  pfm_top_i/dynamic_region/.*/inst/.*/read_edge_list_ptr_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_0__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_0__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_1__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_1__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_2__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_2__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_3__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_3__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_13__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_13__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_14__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_14__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_15__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_15__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/vec_X__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/vec_X__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ptr__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ptr__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_0/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_1/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_2/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_3/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_4/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_0/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_1/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_2/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_3/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_0/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_13/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_14/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_15/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_1/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_2/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_3/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_0/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_1/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_2/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_3/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_4/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_0/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_1/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_2/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_3/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_0/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_1/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_0/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_1/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_2/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_3/.*.unit
} ]
add_cells_to_pblock [get_pblocks CR_X0Y0_To_CR_X3Y3] [get_cells -regex {
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_2
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_2__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_3
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_3__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/FloatvAddFloatv_0
  pfm_top_i/dynamic_region/.*/inst/.*/FloatvAddFloatv_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/FloatvMultConst_0
  pfm_top_i/dynamic_region/.*/inst/.*/FloatvMultConst_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/FloatvMultConst_1
  pfm_top_i/dynamic_region/.*/inst/.*/FloatvMultConst_1__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/Merger_Y_0
  pfm_top_i/dynamic_region/.*/inst/.*/Merger_Y_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_4
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_4__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_5
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_5__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_6
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_6__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_7
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_7__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_8
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_8__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_9
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_9__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_10
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_10__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_4
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_4__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_6
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_6__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_7
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_7__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_4
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_4__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_5
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_5__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_6
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_6__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_7
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_7__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_8
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_8__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_9
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_9__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_10
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_10__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_11
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_11__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_12
  pfm_top_i/dynamic_region/.*/inst/.*/read_A_12__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/read_Y_0
  pfm_top_i/dynamic_region/.*/inst/.*/read_Y_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/write_Y_0
  pfm_top_i/dynamic_region/.*/inst/.*/write_Y_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_4__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_4__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_5__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_5__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_6__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_6__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_7__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_7__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_8__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_8__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_9__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_9__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_10__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_10__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_11__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_11__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_12__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/edge_list_ch_12__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/vec_Y__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/vec_Y__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/vec_Y_out__m_axi
  pfm_top_i/dynamic_region/.*/inst/.*/vec_Y_out__m_axi__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_10/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_11/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_4/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_5/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_6/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_7/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_8/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_9/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_10/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_4/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_5/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_6/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_7/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_8/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_9/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_10/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_11/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_12/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_4/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_5/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_6/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_7/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_8/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_9/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_10/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_11/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_4/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_5/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_6/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_7/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_8/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_9/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_AX/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_alpha_AX/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_in/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_in_beta/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_out/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_4/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_5/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_6/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_7/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_0/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_1/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_2/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_3/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_4/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_5/inst\\[2]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_6/inst\\[2]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_7/inst\\[2]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_10/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_4/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_5/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_6/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_7/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_8/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_9/inst\\[0]\\.unit
} ]
add_cells_to_pblock [get_pblocks CR_X0Y4_To_CR_X3Y7] [get_cells -regex {
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_4
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_4__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_11
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_11__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_12
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_12__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_5
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_5__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_8
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_8__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_9
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_9__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_11/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_12/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_13/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_10/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_11/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_12/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_5/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_8/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_9/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_11/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_12/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_11/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_12/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_13/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_5/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_8/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_9/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_4/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_5/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_6/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_7/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_10/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_11/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_12/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_5/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_8/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_9/inst\\[1]\\.unit
} ]
add_cells_to_pblock [get_pblocks CR_X0Y8_To_CR_X3Y11] [get_cells -regex {
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_5
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_5__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_10
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_10__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_11
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_11__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_10/inst\\[2]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_11/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_10/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_11/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_5/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_10/inst\\[2]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_11/inst\\[1]\\.unit
} ]
add_cells_to_pblock [get_pblocks CR_X4Y4_To_CR_X7Y7] [get_cells -regex {
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_6
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_6__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_7
  pfm_top_i/dynamic_region/.*/inst/.*/Arbiter_Y_7__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_13
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_13__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_14
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_14__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_15
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Xvec_15__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_12
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_12__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_13
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_13__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_14
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_14__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_15
  pfm_top_i/dynamic_region/.*/inst/.*/PEG_Yvec_15__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/black_hole_float_v16_0
  pfm_top_i/dynamic_region/.*/inst/.*/black_hole_float_v16_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/black_hole_int_0
  pfm_top_i/dynamic_region/.*/inst/.*/black_hole_int_0__state.*
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_13/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_14/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_15/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/PE_inst_16/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_12/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_13/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_14/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/Yvec_inst_15/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_13/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_14/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_A_15/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_13/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_14/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_15/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_X_pe_16/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_12/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_13/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_14/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_15/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_6/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_Y_pe_abd_7/inst\\[0]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_12/inst\\[1]\\.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_13/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_14/.*.unit
  pfm_top_i/dynamic_region/.*/inst/.*/fifo_aXvec_15/.*.unit
} ]
foreach pblock [get_pblocks -regexp CR_X\\d+Y\\d+_To_CR_X\\d+Y\\d+] {
  if {[get_property CELL_COUNT $pblock] == 0} {
    puts "WARNING: delete empty pblock $pblock "
    delete_pblocks $pblock
  }
}
foreach pblock [get_pblocks] {
  report_utilization -pblocks $pblock
}