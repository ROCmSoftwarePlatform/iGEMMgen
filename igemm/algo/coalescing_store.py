################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# 
################################################################################

import math
from .igemm_base import *
from .shared_memory import *
from .global_memory import *
from .thread_mapping import *
from .xdlops_mapping import *
import copy
import itertools

IGEMM_COALESCING_GEMM_M_ORDER_M0_M1 = 0
IGEMM_COALESCING_GEMM_M_ORDER_M1_M0 = 1

class ctrl_coalescing_store_t(object):
    def __init__(self):
        self.ctm = None # ctrl_thread_mapping_t
        self.coalescing_groups = 1
        self.block_size = 256
        self.vector_store_size = 1
        self.data_byte = 1
        self.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M0_M1
        self.gemm_m_m0_m1 = []
        # for simplicity, if order is IGEMM_COALESCING_GEMM_M_ORDER_M1_M0, coalescing_groups must be multiply of t_m0
        # use adjust_optimal_coalescing_groups() to do this.

        # self.s_unmerge_cluster_stride_m0 = None     # if not none, means M0*M1 is not continous

    def adjust_optimal_coalescing_groups(self):
        '''
        if coalescing_groups is not multiply of t_m0, we will be very hard to calculate thread stride after load from LDS
        this can help make easy index calculation, performance should be no impact
        '''
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M1_M0:
            cg = self.coalescing_groups
            while cg % self.ctm.t_m0() != 0:
                cg = cg * 2
            assert cg <= self.get_length_m_groups()
            self.coalescing_groups = cg

    def get_length_m_groups(self):
        ''' vgprs per thread in m dimension '''
        return self.ctm.t_mr() * self.ctm.t_m1() * self.ctm.t_m0()

    def get_length_n_groups(self):
        ''' vgprs per thread in n dimension '''
        return self.ctm.t_nr() * self.ctm.t_n1() * self.ctm.t_n0()

    def get_subgroups(self):
        # num_m_group * num_n_group = num_group
        l_mg = self.get_length_m_groups()
        l_ng = self.get_length_n_groups()
        num_m_groups = math.gcd(self.coalescing_groups, l_mg)
        num_n_groups = math.gcd(self.coalescing_groups//num_m_groups, l_ng)
        g_mr = math.gcd(num_m_groups, self.ctm.t_mr())
        g_m1 = math.gcd(num_m_groups//g_mr, self.ctm.t_m1())
        g_m0 = math.gcd(num_m_groups//(g_mr*g_m1), self.ctm.t_m0())
        g_nr = math.gcd(num_n_groups, self.ctm.t_nr())
        g_n1 = math.gcd(num_n_groups//g_nr, self.ctm.t_n1())
        g_n0 = math.gcd(num_n_groups//(g_nr*g_n1), self.ctm.t_n0())
        return g_mr, g_m1, g_m0, g_nr, g_n1, g_n0   # g_m1 always 1

    def get_transposed_thread_mapping(self):
        # after coalescing, the thread mapping is indeed transposed
        assert self.ctm.t_n0() % self.vector_store_size == 0
        assert self.block_size == (self.ctm.c_mr()*self.ctm.c_nr()*self.ctm.c_m0()*self.ctm.c_n0()*self.ctm.c_m1()*self.ctm.c_n1())
        n_n_total = self.ctm.n_n_total()
        n_m_total = self.ctm.n_m_total()
        assert n_n_total % self.vector_store_size == 0
        #trans_t_mr, trans_t_nr, trans_t_m1, trans_t_m0, trans_t_n1, trans_t_n0 = 1,1,1,1,1,1
        #trans_c_mr, trans_c_nr, trans_c_m1, trans_c_m0, trans_c_n1, trans_c_n0 = 1,1,1,1,1,1

        trans_t_n0 = self.vector_store_size
        trans_c_n0 = n_n_total // trans_t_n0
        trans_t_m0 = 1
        trans_c_m0 = self.block_size // trans_c_n0
        trans_t_n1 = 1
        trans_c_n1 = 1
        trans_t_m1 = n_m_total // (trans_t_m0 * trans_c_m0)
        trans_c_m1 = 1
        trans_t_nr = 1
        trans_c_nr = 1
        trans_t_mr = 1
        trans_c_mr = 1

        transposed_thread_mapping = ctrl_thread_mapping_t()
        transposed_thread_mapping.thread_lengths =  [trans_t_mr, trans_t_nr, trans_t_m1, trans_t_n1, trans_t_m0, trans_t_n0]
        transposed_thread_mapping.cluster_lengths = [trans_c_mr, trans_c_nr, trans_c_m1, trans_c_n1, trans_c_m0, trans_c_n0]

        return transposed_thread_mapping

    def get_thread_m_stride(self):
        ttm = self.get_transposed_thread_mapping()
        assert ttm.t_m0() == 1
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.get_subgroups()
        assert g_m1 == 1

        # do some assert
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            m_index_per_group = self.get_m_index_per_group()
            thread_m_stride = g_m0 * ttm.n_m0()
        else:
            m_index_per_group = self.get_m_index_per_group_m1_m0()
            thread_m_stride = -1
        for ig in range(len(m_index_per_group)):
            for ic in range(len(m_index_per_group[ig])):
                _list = m_index_per_group[ig][ic]
                for idx in range(len(_list) - 1):
                    diff_m = _list[idx+1] - _list[idx]
                    if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
                        assert diff_m == thread_m_stride, f"diff_m:{diff_m}, thread_m_stride:{thread_m_stride}"
                    else:
                        if thread_m_stride == -1:
                            thread_m_stride = diff_m
                        assert diff_m == thread_m_stride, f"diff_m:{diff_m}, thread_m_stride:{thread_m_stride}"

        return thread_m_stride

    def get_num_dword_per_group(self):
        return (self.ctm.t_mr()*self.ctm.t_nr()*self.ctm.t_m0()*self.ctm.t_n0()*self.ctm.t_m1()*self.ctm.t_n1()) // self.coalescing_groups

    def get_sub_m0_offset(self, i_c_m0):
        ttm = self.get_transposed_thread_mapping()
        assert ttm.t_m0() == 1
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.get_subgroups()
        assert g_m1 == 1
        sub_m0_offset = ((i_c_m0 >> int(math.log2(g_m0))) << self.ctm.t_m0()) | (i_c_m0 & (g_m0 - 1))
        return sub_m0_offset
        #print(" i_c_m0 >> igemm_log2(g_m0) << self.t_m0   i_c_m0 & (g_m0 - 1)  ")

    def get_m0_m1_index(self, m_index):
        assert len(self.gemm_m_m0_m1) != 0
        m0, m1 = self.gemm_m_m0_m1[0], self.gemm_m_m0_m1[1]
        # print(f"m0:{m0}, m1:{m1}")
        assert m_index < m0 * m1, f"m_index:{m_index} larger than gemm_m_m0_m1:{self.gemm_m_m0_m1}, please check"
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_index // m1, m_index % m1
        else:
            return m_index % m0, m_index // m0

    def get_m_index_per_group(self):
        num_dword_per_group = self.get_num_dword_per_group()
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.get_subgroups()
        assert g_nr == 1 and g_n1 == 1 and g_n0 == 1
        assert g_m1 == 1
        l_mr = self.ctm.t_mr() // g_mr
        l_m0 = self.ctm.t_m0() // g_m0
        ttm = self.get_transposed_thread_mapping()
        m_index_per_group = list()
        for i_g_mr in range(g_mr):
            for i_g_m0 in range(g_m0):
                m_idx_start_per_group = i_g_mr * l_mr * self.ctm.n_m0()*self.ctm.n_m1() + i_g_m0 * l_m0
                # first, generate current group, m index
                m_index = []
                for t_mr in range(l_mr):
                    for tid_along_m0 in range(self.ctm.c_m0()):
                        for tid_along_m1 in range(self.ctm.c_m1()):
                            m_idx_start_per_m0 = m_idx_start_per_group + tid_along_m1 * self.ctm.n_m0() + tid_along_m0 * self.ctm.t_m0()
                            for t_m0 in range(l_m0):
                                m_idx_start = m_idx_start_per_m0 + t_mr*self.ctm.n_m0()*self.ctm.n_m1() + t_m0
                                m_index.append(m_idx_start)
                assert len(m_index) == ttm.c_m0() * num_dword_per_group, f"{len(m_index)}, {ttm.c_m0()}, {num_dword_per_group}"
                m_index.sort()
                #print(f"xxxxx:{m_index}")
                pixel_m_index = [None] * ttm.c_m0()
                # second, record index according to current id
                for i, m in enumerate(m_index):
                    _icm0 = i % ttm.c_m0()
                    if not pixel_m_index[_icm0]:
                        pixel_m_index[_icm0] = list()
                    pixel_m_index[_icm0].append(m)
                m_index_per_group.append(pixel_m_index)
        return m_index_per_group

    def get_m_index_from_m1_m0(self, m_idx):
        assert len(self.gemm_m_m0_m1) == 2
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_idx
        n_m0, n_m1 = self.gemm_m_m0_m1[0], self.gemm_m_m0_m1[1]
        i_m0 = m_idx % n_m0
        i_m1 = m_idx // n_m0
        return i_m0 * n_m1 + i_m1

    def get_m_index_per_group_m1_m0(self):
        m_index_per_group = copy.copy(self.get_m_index_per_group())
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_index_per_group
        assert len(self.gemm_m_m0_m1) == 2
        assert len(m_index_per_group) == self.coalescing_groups
        for ig in range(len(m_index_per_group)):
            for i_cm0 in range(len(m_index_per_group[ig])):
                for i_t in range(len(m_index_per_group[ig][i_cm0])):
                    m_index = m_index_per_group[ig][i_cm0][i_t]
                    m_index_per_group[ig][i_cm0][i_t] = self.get_m_index_from_m1_m0(m_index)
        return m_index_per_group

class igemm_coalescing_store_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_t
        assert ctrl.ctm.t_mr() == 2 and ctrl.ctm.t_nr() == 2
        assert ctrl.block_size % ctrl.ctm.n_n_total() == 0, f"block_size:{ctrl.block_size}, gemm_n:{ctrl.ctm.n_n_total()}"
        self.ctrl = ctrl

    def name(self):
        return ''

    def init_co_lds_offset(self, v_co_sst, v_co_sld, v_gemm_im, v_gemm_in, v_tid, v_tmp2):
        '''
        v_gemm_im, v_gemm_in must be computed from get_gemm_index_for_dst_matrix
        '''
        ctrl = self.ctrl
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.ctrl.get_subgroups()
        with self._deferred_context():
            self._emit(f"; init_co_lds_offset")
            gemm_m_shrink = g_m0
            if gemm_m_shrink != 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}], {igemm_log2(gemm_m_shrink)}, v[{v_gemm_im}]")
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp2}], {igemm_log2(ctrl.ctm.n_m_total())}, v[{v_gemm_in}]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_gemm_im}], {igemm_log2(ctrl.ctm.n_m_total())}, v[{v_gemm_in}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {igemm_log2(ctrl.data_byte)}, v[{v_co_sst}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sld}], {igemm_log2(ctrl.data_byte * ctrl.vector_write_out)}, v[{v_tid}]")

        return self._get_deferred()


    def init_co_sub_m_index(self, v_co_sub_m_index, v_tid, v_tmp2):
        ctrl = self.ctrl
        # need use v_co_sub_m_index to calculate v offset in m direction
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = ctrl.get_subgroups()
        assert g_m1 == 1
        # sub_m0_offset = ((i_c_m0 >> int(math.log2(g_m0))) << self.ctm.t_m0()) | (i_c_m0 & (g_m0 - 1))
        l_mr = ctrl.ctm.t_mr() // g_mr
        l_m0 = ctrl.ctm.t_m0() // g_m0

        with self._deferred_context():
            self._emit(f"; init_co_sub_m_index")
            if ctrl.vector_write_out == 1:
                self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(ctrl.ctm.n_n_total())}, v[{v_tid}]")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {igemm_log2(ctrl.vector_write_out)}, v[{v_tid}]")
                self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(ctrl.ctm.n_n_total())}, v[{v_tmp2}]")

            self._emit(f"v_and_b32 v[{v_tmp2}], {l_m0 - 1}, v[{v_co_sub_m_index}]")
            self._emit(f"v_lshrrev_b32 v[{v_tmp2}+1], {igemm_log2(l_m0)}, v[{v_co_sub_m_index}]")
            self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}] v[{v_tmp2}+1], {igemm_log2(ctrl.ctm.t_m0())}, v[{v_tmp2}]")
        return self._get_deferred()

    def init_co_sub_n_index(self, v_co_sub_n_index, v_tid, v_tmp2):
        ctrl = self.ctrl
        # need use v_co_sub_n_index to calculate v offset in n direction
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = ctrl.get_subgroups()
        assert g_m1 == 1
        l_mr = ctrl.ctm.t_mr() // g_mr
        l_m0 = ctrl.ctm.t_m0() // g_m0

        with self._deferred_context():
            self._emit(f"; init_co_sub_n_index")
            if ctrl.vector_write_out == 1:
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.ctm.n_n_total() - 1}, v[{v_tid}]")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {igemm_log2(ctrl.vector_write_out)}, v[{v_tid}]")
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.ctm.n_n_total() - 1}, v[{v_tmp2}]")
        return self._get_deferred()


    def __call__(self, v_c, v_co_sst, v_co_sld, s_p_out, v_out_offset, s_out_offset, s_gemm_m0_stride, s_gemm_m1_stride, s_tmp4, v_store_flag = None):
        # if no need s_out_offset, set to integer 0
        # if no need flag to dicide store, set v_store_flag to 0
        ctrl = self.ctrl
        v_c = sym_t(v_c)
        v_co_sst = sym_t(v_co_sst)
        v_co_sld = sym_t(v_co_sld)
        s_tmp4 = sym_t(s_tmp4)
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.ctrl.get_subgroups()
        assert g_m1 == 1 and g_nr == 1 and g_n1 == 1 and g_n0 == 1
        l_mr = ctrl.ctm.t_mr() // g_mr
        l_m0 = ctrl.ctm.t_m0() // g_m0
        no_s_out_offset = s_out_offset is None

        # mc, vec_count, vec_byte, vec_stride, sst_base=0):
        inst_sst = inst_ds_write2_likely_t(self.mc, 2, ctrl.ctm.t_n0() * ctrl.data_byte, ctrl.ctm.n_n_total() * ctrl.data_byte // 2)
        # mc, vec_count, vec_byte, vec_stride, sld_base = 0):
        inst_sld = inst_ds_read2_likely_t(self.mc, 2, ctrl.vector_write_out * ctrl.data_byte, ctrl.block_size * ctrl.vector_write_out * ctrl.data_byte)
        # self, vdata, vaddr, srsrc, soffset, offset):
        inst_gst = inst_buffer_store_dword_t(ctrl.vector_write_out)

        s_out_offset_itr = sym_t(s_tmp4(0))
        s_thread_m_stride = sym_t(s_tmp4(1))

        if ctrl.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            m_index_per_group = ctrl.get_m_index_per_group()
        else:
            m_index_per_group = ctrl.get_m_index_per_group_m1_m0()
        thread_m_stride = ctrl.get_thread_m_stride()

        assert len(m_index_per_group) == ctrl.coalescing_groups


        with self._deferred_context():
            self._emit(f"; coalescing store, gemm_mxn:{ctrl.ctm.n_m_total()}x{ctrl.ctm.n_n_total()}, block:{ctrl.block_size}, m_repeatxm_perthread:{ctrl.ctm.t_mr()}x{ctrl.ctm.t_m0()}, n_repeatxn_perthread:{ctrl.ctm.t_nr()}x{ctrl.ctm.t_n0()}")
            self._emit(f"; coalescing_groups:{ctrl.coalescing_groups}, num_dword_per_group:{ctrl.get_num_dword_per_group()}")
            self._emit(f"; coalescing_groups_in_m_repeat:{g_mr}, group_length_in_m_repeat:{l_mr}, coalescing_groups_in_m_per_thread:{g_m0}, group_length_in_m_per_thread:{l_m0}")
            # emit some pre index
            if ctrl.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M1_M0 and s_gemm_m0_stride is not None:
                self._emit(f"s_mul_i32 s[{s_thread_m_stride()}], {thread_m_stride}, s[{s_gemm_m0_stride}]    ; init per thread stride in m dimension")
            else:
                self._emit(f"s_mul_i32 s[{s_thread_m_stride()}], {thread_m_stride}, s[{s_gemm_m1_stride}]    ; init per thread stride in m dimension")

            for i_group in range(ctrl.coalescing_groups):
                m_index_start_per_group = m_index_per_group[i_group][0][0]
                m0_index_start_per_group, m1_index_start_per_group = ctrl.get_m0_m1_index(m_index_start_per_group)

                c_group_start_index = i_group * l_mr * l_m0 * ctrl.ctm.t_n0() * ctrl.ctm.t_nr()
                current_m_index = m_index_per_group[i_group]
                self._emit(f"; start group {i_group}, m index start from {m_index_start_per_group}")
                self._emit(f"s_barrier")
                for i_sub_length in range(l_mr * l_m0):
                    c_sub_start_index = c_group_start_index + i_sub_length * ctrl.ctm.t_n0() * ctrl.ctm.t_nr()
                    sst_offset = i_sub_length * ctrl.ctm.n_n_total() * ctrl.data_byte
                    self._emit(inst_sst(v_co_sst(), v_c(c_sub_start_index), sst_offset))

                self._emit(f"s_waitcnt lgkmcnt(0)")
                self._emit(f"s_barrier")
                self._emit(f";   load from lds")
                issue_list = []
                for i_d in range(ctrl.get_num_dword_per_group() // (2 * ctrl.vector_write_out)):
                    c_sub_start_index = c_group_start_index + i_d * 2 * ctrl.vector_write_out
                    sld_offset = i_d * 2 * ctrl.block_size * ctrl.vector_write_out * ctrl.data_byte
                    self._emit(inst_sld(v_c(c_sub_start_index), v_co_sld(), sld_offset))
                    issue_list.append(inst_sld.get_issues(sld_offset))

                if v_store_flag is not None and type(v_store_flag) is str:
                    self._emit(f"v_cmpx_eq_u32 vcc, 1, v[{v_store_flag}]")
                    #self._emit(f"s_cbranch_execz {label_prefix}_co_{i_group}")

                self._emit(f";   store to global, m index start from {m_index_start_per_group}, m0:{m0_index_start_per_group}, m1:{m1_index_start_per_group}")
                if s_gemm_m0_stride is not None:
                    self._emit(f"s_mul_i32 s[{s_tmp4(2)}], {m0_index_start_per_group}, s[{s_gemm_m0_stride}]")
                    self._emit(f"s_mul_i32 s[{s_tmp4(3)}], {m1_index_start_per_group}, s[{s_gemm_m1_stride}]")
                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp4(2)}], s[{s_tmp4(3)}]")
                else:
                    if m_index_start_per_group == 0:
                        if no_s_out_offset:
                            self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], 0")
                        else:
                            self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_out_offset}]")
                    elif m_index_start_per_group == 1:
                        if no_s_out_offset:
                            self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}]")
                        else:
                            self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}], s[{s_out_offset}]")
                    else:
                        if no_s_out_offset:
                            self._emit(f"s_mul_i32 s[{s_out_offset_itr()}], {m_index_start_per_group}, s[{s_gemm_m1_stride}]")
                        else:
                            self._emit(f"s_mul_i32 s[{s_tmp4(3)}], {m_index_start_per_group}, s[{s_gemm_m1_stride}]")
                            self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp4(3)}], s[{s_out_offset}]")
                i_m0_start, i_m1_start =  m0_index_start_per_group, m1_index_start_per_group
                for i_gst in range(ctrl.get_num_dword_per_group() // ctrl.vector_write_out):
                    if i_gst % 2 == 0:
                        i_issues =  (i_gst // 2) + 1
                        i_issue_list = issue_list[i_issues:]
                        i_issue_cnt = igemm_flatten_list_accumulate(i_issue_list) if len(i_issue_list) != 0 else 0
                        self._emit(f"s_waitcnt lgkmcnt({i_issue_cnt})")
                    # vdata, vaddr, srsrc, soffset, offset
                    self._emit(inst_gst(v_c(c_group_start_index + i_gst*ctrl.vector_write_out), v_out_offset, s_p_out, s_out_offset_itr(), 0))
                    if i_gst != (ctrl.get_num_dword_per_group() // ctrl.vector_write_out) - 1:
                        if s_gemm_m0_stride is not None:
                            i_m = m_index_per_group[i_group][0][i_gst+1]
                            i_m0, i_m1 = ctrl.get_m0_m1_index(i_m)
                            self._emit(f"; im:{i_m}, i_m0:{i_m0}, i_m1:{i_m1}")
                            if ctrl.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
                                if i_m0 > i_m0_start:
                                    i_m0_start = i_m0
                                    # m0 accumulate
                                    self._emit(f"s_mul_i32 s[{s_tmp4(2)}], {i_m0}, s[{s_gemm_m0_stride}]")
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp4(2)}], s[{s_tmp4(3)}]")
                                else:
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_thread_m_stride()}], s[{s_out_offset_itr()}]")
                            else:
                                if i_m1 > i_m1_start:
                                    i_m1_start = i_m1
                                    # m1 accumllate
                                    self._emit(f"s_mul_i32 s[{s_tmp4(3)}], {i_m1}, s[{s_gemm_m1_stride}]")
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp4(2)}], s[{s_tmp4(3)}]")
                                else:
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_thread_m_stride()}], s[{s_out_offset_itr()}]")
                        else:
                            self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_thread_m_stride()}], s[{s_out_offset_itr()}]")
                if v_store_flag is not None and type(v_store_flag) is str:
                    self._emit(f"s_mov_b64 exec, -1")
        return self._get_deferred()

class ctrl_coalescing_store_xdlops_t(object):
    def __init__(self):
        self.cxm = None # ctrl_xdlops_mapping_t
        self.coalescing_groups = 1
        self.block_size = 256
        self.vector_store_size = 1
        self.data_byte = 1
        self.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M0_M1
        self.gemm_m_m0_m1 = []
        self.gemm_k_global_split = False

    def adjust_optimal_coalescing_groups(self):
        '''
        in xdlops M1_M0 order if have better write pattern, change cgroup is quite complex.
        '''
        # if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M1_M0:
        #     cg = self.coalescing_groups
        #     while cg % self.ctm.t_m0() != 0:
        #         cg = cg * 2
        #     assert cg <= self.get_length_m_groups()
        #     self.coalescing_groups = cg
        pass

    def can_skip_coalescing(self):
        '''
        in some configuration, after LDS shuffle the data layout is the same compare with un-shuffled shape.
        '''
        # if self.coalescing_groups == 1 and self.cxm.wave_step_m == 1 and self.cxm.wave_step_n == 1 and \
        #         self.cxm.wave_repeat_m == 1 and self.cxm.wave_repeat_n == 1:
        #     if self.cxm.wave_tile_m == self.cxm.inst_mfma.m:
        #         return True
        #     if self.cxm.wave_tile_n == self.cxm.macro_tile_n:
        #         return True
        return False

    def get_length_m_groups(self):
        ''' agpr per thread in m dimension '''
        return self.cxm.wave_repeat_m * self.cxm.wave_step_m * self.cxm.lanegroup_m_per_wave() * self.cxm.lanegroup_m_per_block() * self.cxm.lanegroup_m_per_thread()

    def get_length_n_groups(self):
        ''' agpr per thread in n dimension '''
        return self.cxm.wave_repeat_n * self.cxm.wave_step_n * self.cxm.lanegroup_n_per_wave() * self.cxm.lanegroup_n_per_block() * self.cxm.lanegroup_n_per_thread() # xdlops in n dimension always 1

    def get_length_m_max_groups(self):
        ''' maximum number of agpr along m dimension per thread can be divided into groups.
            but consider LDS load/store, we want to utilize the fact that every lanegroup is a 4x64 matrix,
            which might be distributed along multiple blocks (like 4x4x1), or every block contains multiple lanegroups(like 16x16x1, 32x32x1)
            henve in m dimension there always have a granularity of 4 per thread, that is continuous along m dimension.
            we want to use a single ds_write_b128/ds_read_b128 to deal with this granularity (which implies a transpose)
            hence we do not want split inside the "4" granularity within a thread
        '''
        return self.get_length_m_groups() // AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M

    def get_subgroups(self):
        def split_ndim_length(total_length, dim_lengths):
            assert type(dim_lengths) in (list, tuple)
            length = total_length
            split_length = list()
            for d in dim_lengths:
                s = math.gcd(d, length)
                length = length // s
                split_length.append(s)
            return tuple(split_length)

        # assert self.get_length_m_groups() % self.coalescing_groups == 0, \
        #     f"if coalescing groups:{self.coalescing_groups} larger than single xdlops agpr number along m:{self.get_length_m_groups()}, can not do this split"
        assert self.get_length_m_max_groups() % self.coalescing_groups == 0, \
            f"if coalescing groups:{self.coalescing_groups} larger than maximum single xdlops agpr(divided by 4) number along m:{self.get_length_m_max_groups()}, can not do this split"
        l_mg = self.get_length_m_groups()
        l_ng = self.get_length_n_groups()
        num_m_groups = math.gcd(self.coalescing_groups, l_mg)
        num_n_groups = math.gcd(self.coalescing_groups//num_m_groups, l_ng)

        assert num_n_groups == 1, "if have multiple groups along n dimension, coalesing is meaningless"

        # for lanegroup_per_cluster, since within thread will not contain this, so must specify this to 1
        split_m_lengths = (self.cxm.wave_repeat_m, self.cxm.wave_step_m, \
                 self.cxm.lanegroup_m_per_wave(), self.cxm.lanegroup_m_per_block(), self.cxm.lanegroup_m_per_thread())
        g_mr, g_ms, g_mw, g_mb, g_mt = split_ndim_length(num_m_groups, split_m_lengths)
        assert g_mt == 1, 'we do not want to split inside this granularity within a thread'

        return g_mr, g_ms, g_mw, g_mb, g_mt # groups in m_repeat, m_step, lanegroup_m_per_wave, lanegroup_m_per_block, lanegroup_m_per_thread

    def get_subgroup_length(self):
        g_mr, g_ms, g_mw, g_mb, g_mt = self.get_subgroups()
        # self.cxm.wave_repeat_m, self.cxm.wave_step_m, self.cxm.lanegroup_m_per_wave(), self.cxm.lanegroup_m_per_block(), self.cxm.lanegroup_m_per_thread()

        l_mr = self.cxm.wave_repeat_m // g_mr
        l_ms = self.cxm.wave_step_m // g_ms
        l_mw = self.cxm.lanegroup_m_per_wave() // g_mw
        l_mb = self.cxm.lanegroup_m_per_block() // g_mb
        l_mt = self.cxm.lanegroup_m_per_thread() // g_mt
        return l_mr, l_ms, l_mw, l_mb, l_mt

    def get_transposed_thread_mapping(self):
        # xdlops need transfer agpr to vgpr, then do LDS shuffle
        # here we still use legacy thread mapping to describe
        assert self.vector_store_size == 1

        n_n_total = self.cxm.macro_tile_n
        n_m_total = self.cxm.macro_tile_m
        assert self.cxm.waves * AMDGPU_WAVE_SIZE % n_n_total == 0, f"waves:{self.cxm.waves}, n_n_total:{n_n_total}, cxm:{self.cxm.serialize()}"

        trans_t_n0 = self.vector_store_size
        trans_c_n0 = n_n_total // trans_t_n0
        trans_t_m0 = AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M  # here we append granularity to perthread m0
        trans_c_m0 = self.block_size // trans_c_n0
        trans_t_n1 = 1
        trans_c_n1 = 1
        assert n_m_total % (trans_t_m0 * trans_c_m0) == 0
        trans_t_m1 = n_m_total // (trans_t_m0 * trans_c_m0)
        trans_c_m1 = 1
        trans_t_nr = 1
        trans_c_nr = 1
        trans_t_mr = 1
        trans_c_mr = 1

        transposed_thread_mapping = ctrl_thread_mapping_t()
        transposed_thread_mapping.thread_lengths =  [trans_t_mr, trans_t_nr, trans_t_m1, trans_t_n1, trans_t_m0, trans_t_n0]
        transposed_thread_mapping.cluster_lengths = [trans_c_mr, trans_c_nr, trans_c_m1, trans_c_n1, trans_c_m0, trans_c_n0]

        return transposed_thread_mapping

    #def get_thread_m_stride(self):
    #    ttm = self.get_transposed_thread_mapping()
    #    assert ttm.t_m0() == 1
    #    g_mr, g_ms, g_mw, g_mb, g_mt = self.get_subgroups()
    #    l_mr, l_ms, l_mw, l_mb, l_mt = self.get_subgroup_length()

    #    # do some assert
    #    if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
    #        m_index_per_group = self.get_m_index_per_group()
    #        thread_m_stride = g_m0 * ttm.n_m0()
    #    else:
    #        m_index_per_group = self.get_m_index_per_group_m1_m0()
    #        thread_m_stride = -1
    #    for ig in range(len(m_index_per_group)):
    #        for ic in range(len(m_index_per_group[ig])):
    #            _list = m_index_per_group[ig][ic]
    #            for idx in range(len(_list) - 1):
    #                diff_m = _list[idx+1] - _list[idx]
    #                if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
    #                    assert diff_m == thread_m_stride, f"diff_m:{diff_m}, thread_m_stride:{thread_m_stride}"
    #                else:
    #                    if thread_m_stride == -1:
    #                        thread_m_stride = diff_m
    #                    assert diff_m == thread_m_stride, f"diff_m:{diff_m}, thread_m_stride:{thread_m_stride}"

    #    return thread_m_stride

    def get_num_dword_per_group(self):
        assert self.cxm.total_acc_c() % self.coalescing_groups == 0, \
                f"total_acc_c:{self.cxm.total_acc_c()}, coalescing_groups:{self.coalescing_groups}, m_groups:{self.get_length_m_groups()}, inst:{self.cxm.inst_mfma.m}x{self.cxm.inst_mfma.n}x{self.cxm.inst_mfma.k}"
        return self.cxm.total_acc_c() // self.coalescing_groups

    def get_sub_m0_offset(self, i_c_m0):
        ttm = self.get_transposed_thread_mapping()
        assert ttm.t_m0() == 1
        g_mr, g_ms, g_m0, g_nr, g_ns, g_n0 = self.get_subgroups()
        assert g_m1 == 1
        sub_m0_offset = ((i_c_m0 >> int(math.log2(g_m0))) << self.ctm.t_m0()) | (i_c_m0 & (g_m0 - 1))
        return sub_m0_offset
        #print(" i_c_m0 >> igemm_log2(g_m0) << self.t_m0   i_c_m0 & (g_m0 - 1)  ")

    def get_m0_m1_index(self, m_index):
        assert len(self.gemm_m_m0_m1) != 0
        m0, m1 = self.gemm_m_m0_m1[0], self.gemm_m_m0_m1[1]
        # print(f"m0:{m0}, m1:{m1}")
        assert m_index < m0 * m1, f"m_index:{m_index} larger than gemm_m_m0_m1:{self.gemm_m_m0_m1}, please check"
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_index // m1, m_index % m1
        else:
            return m_index % m0, m_index // m0

    def get_m_index_per_group(self):
        '''
        get m index after LDS shuffle
        '''
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        num_dword_per_group = self.get_num_dword_per_group()
        g_mr, g_ms, g_mw, g_mb, g_mt = self.get_subgroups()
        l_mr, l_ms, l_mw, l_mb, l_mt = self.get_subgroup_length()

        assert num_dword_per_group == (l_mr * l_ms * l_mw * l_mb * l_mt) * flatten(self.cxm.acc_c_per_thread_n()), \
                f"num_dword_per_group:{num_dword_per_group}, l_mr:{l_mr}, l_ms:{l_ms}, l_mw:{l_mw}, l_mb:{l_mb}, l_mt:{l_mt} x acc_c_per_thread_n:{self.cxm.acc_c_per_thread_n()}"

        # print(f"mr:{g_mr}x{l_mr}x{self.cxm.wave_repeat_m}, ms:{g_ms}x{l_ms}x{self.cxm.wave_step_m}, mw:{g_mw}x{l_mw}x{self.cxm.lanegroup_m_per_wave()}, mb:{g_mb}x{l_mb}x{self.cxm.lanegroup_m_per_block()}, mt:{g_mt}x{l_mt}x{self.cxm.lanegroup_m_per_thread()}")

        ttm = self.get_transposed_thread_mapping()
        m_index_per_group = list()
        m_index_total = list()
        # for i_g_mr in range(g_mr):
        #     for i_g_ms in range(g_ms):
        #         for i_g_mw in range(g_mw):
        #             for i_g_mb in range(g_mb):
        #                 for i_g_mt in range(g_mt):
        #                     m_idx_start_per_group = i_g_mr * l_mr * self.cxm.wave_step_m * self.cxm.wave_tile_m * self.cxm.waves_per_m() + \
        #                                     i_g_ms * l_ms * self.cxm.wave_tile_m + \
        #                                     i_g_mw * l_mw * self.cxm.lanegroup_m_per_block() * self.cxm.lanegroup_m_per_cluster() * self.cxm.lanegroup_m_per_thread() + \
        #                                     i_g_mb * l_mb * self.cxm.lanegroup_m_per_cluster() * self.cxm.lanegroup_m_per_thread() + \
        #                                     i_g_mt * l_mt
        #                     # print(f"m_idx_start_per_group:{m_idx_start_per_group}")
        #                     m_index = []
        #                     for t_along_waves_per_m in range(self.cxm.waves_per_m()):
        #                         for t_along_block_m_per_lanegroup in range(self.cxm.block_m_per_lanegroup()):
        #                             for t_along_lanegroup_m_per_cluster in range(self.cxm.lanegroup_m_per_cluster()):
        #                                 m_idx_start = m_idx_start_per_group + t_along_waves_per_m * self.cxm.wave_step_m * self.cxm.wave_tile_m + \
        #                                                 t_along_block_m_per_lanegroup * self.cxm.lanegroup_m_per_thread() * self.cxm.lanegroup_m_per_cluster() +\
        #                                                 t_along_lanegroup_m_per_cluster * self.cxm.lanegroup_m_per_thread()
        #                                 for i_mr in range(l_mr):
        #                                     for i_ms in range(l_ms):
        #                                         for i_mw in range(l_mw):
        #                                             for i_mb in range(l_mb):
        #                                                 for i_mt in range(l_mt):
        #                                                     m_idx_current = m_idx_start + i_mr * self.cxm.wave_step_m * self.cxm.wave_tile_m * self.cxm.waves_per_m() + \
        #                                                             i_ms * self.cxm.wave_tile_m + \
        #                                                             i_mw * self.cxm.lanegroup_m_per_block() * self.cxm.lanegroup_m_per_cluster() * self.cxm.lanegroup_m_per_thread() + \
        #                                                             i_mb * self.cxm.lanegroup_m_per_cluster() * self.cxm.lanegroup_m_per_thread() + \
        #                                                             i_mt
        #                                                     m_index.append(m_idx_current)
        #                     m_index.sort()
        #                     m_index_total.extend(m_index)
        #                     pixel_m_index = [None] * ttm.c_m0()
        #                     for i, m in enumerate(m_index):
        #                         _icm0 = (i // ttm.t_m0()) % ttm.c_m0()  # NOTE! here is m granularity take place
        #                         if not pixel_m_index[_icm0]:
        #                             pixel_m_index[_icm0] = list()
        #                         pixel_m_index[_icm0].append(m)
        #                     m_index_per_group.append(pixel_m_index)

        for i_g_mr, i_g_ms, i_g_mw, i_g_mb, i_g_mt in itertools.product(range(g_mr), range(g_ms), range(g_mw), range(g_mb), range(g_mt)):
            m_idx_start_per_group = i_g_mr * l_mr * self.cxm.wave_step_m * self.cxm.wave_tile_m * self.cxm.waves_per_m() + \
                    i_g_ms * l_ms * self.cxm.wave_tile_m + \
                    i_g_mw * l_mw * self.cxm.lanegroup_m_per_block() * self.cxm.lanegroup_m_per_cluster() * self.cxm.lanegroup_m_per_thread() + \
                    i_g_mb * l_mb * self.cxm.lanegroup_m_per_cluster() * self.cxm.lanegroup_m_per_thread() + \
                    i_g_mt * l_mt
            m_index = []
            for t_along_waves_per_m, t_along_block_m_per_lanegroup, t_along_lanegroup_m_per_cluster in itertools.product(
                                    range(self.cxm.waves_per_m()), range(self.cxm.block_m_per_lanegroup()), range(self.cxm.lanegroup_m_per_cluster())):
                m_idx_start = m_idx_start_per_group + t_along_waves_per_m * self.cxm.wave_step_m * self.cxm.wave_tile_m + \
                                t_along_block_m_per_lanegroup * self.cxm.lanegroup_m_per_thread() * self.cxm.lanegroup_m_per_cluster() +\
                                t_along_lanegroup_m_per_cluster * self.cxm.lanegroup_m_per_thread()
                for i_mr, i_ms, i_mw, i_mb, i_mt in itertools.product(range(l_mr), range(l_ms), range(l_mw), range(l_mb), range(l_mt)):
                    m_idx_current = m_idx_start + i_mr * self.cxm.wave_step_m * self.cxm.wave_tile_m * self.cxm.waves_per_m() + \
                            i_ms * self.cxm.wave_tile_m + \
                            i_mw * self.cxm.lanegroup_m_per_block() * self.cxm.lanegroup_m_per_cluster() * self.cxm.lanegroup_m_per_thread() + \
                            i_mb * self.cxm.lanegroup_m_per_cluster() * self.cxm.lanegroup_m_per_thread() + \
                            i_mt
                    m_index.append(m_idx_current)
            m_index.sort()
            m_index_total.extend(m_index)
            pixel_m_index = [None] * ttm.c_m0()
            for i, m in enumerate(m_index):
                _icm0 = (i // ttm.t_m0()) % ttm.c_m0()  # NOTE! here is m granularity take place
                if not pixel_m_index[_icm0]: pixel_m_index[_icm0] = list()
                pixel_m_index[_icm0].append(m)
            m_index_per_group.append(pixel_m_index)

        m_index_total.sort()
        # print(m_index_total)
        assert m_index_total == [xx for xx in range(self.cxm.macro_tile_m)], f"len:{len(m_index_total)}, {m_index_total}"

        # do some validation
        for ig in range(len(m_index_per_group)):
            if len(m_index_per_group[ig]) == 1:
                continue
            for ic in range(len(m_index_per_group[ig]) - 1):
                c_curr_list = m_index_per_group[ig][ic]
                c_next_list = m_index_per_group[ig][ic + 1]
                assert len(c_curr_list) == len(c_next_list)
                diff = c_next_list[0] - c_curr_list[0]
                for i in range(1, len(c_curr_list)):
                    diff_i = c_next_list[i] - c_curr_list[i]
                    assert diff == diff_i, "stride between different transpose m0 not the same, should not happen"

        return m_index_per_group

    def get_co_sub_m_index(self):
        '''
        after LDS shuffle, before store to global, now thread-mapping is inteed transposed (see get_transposed_thread_mapping)
        this function try to get the m_index of the first element of each ttm.c_m0
        '''
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        ttm = self.get_transposed_thread_mapping()
        g_mr, g_ms, g_mw, g_mb, g_mt = self.get_subgroups()
        l_mr, l_ms, l_mw, l_mb, l_mt = self.get_subgroup_length()
        n_mc = self.cxm.lanegroup_m_per_cluster()       # this iteration is among different thread
        n_ml = self.cxm.block_m_per_lanegroup()         # this iteration is among different thread
        n_mv = self.cxm.waves_per_m()                   # this iteration is among different thread
        #print("<+> g_mr:{}, g_ms:{}, g_mw:{}, g_mb:{}, g_mt:{}  ==  l_mr:{}, l_ms:{}, l_mw:{}, l_mb:{}, l_mt:{} | n_mc:{}, n_ml:{}, n_mv:{}".format(
        #        g_mr, g_ms, g_mw, g_mb, g_mt, l_mr, l_ms, l_mw, l_mb, l_mt, n_mc, n_ml, n_mv))
        sub_m_index = [0] * ttm.c_m0()
        # for ic in range(ttm.c_m0()):
        #     nic = ic
        #     x_mc = nic % n_mc; nic = nic // n_mc
        #     x_ml = nic % n_ml; nic = nic // n_ml
        #     x_mb = nic % l_mb; nic = nic // l_mb
        #     x_mw = nic % l_mw; nic = nic // l_mw
        #     x_ms = nic % l_ms; nic = nic // l_ms
        #     x_mv = nic % n_mv; nic = nic // n_mv
        #     x_mr = nic % l_mr
        #     # print("    +-> x_mc:{}, x_ml:{}, x_mb:{}, x_mw:{}, x_ms:{}, x_mv:{}, x_mr:{}".format(x_mc, x_ml, x_mb, x_mw, x_ms, x_mv, x_mr))
        #     sub_m = x_mr * n_mv + x_mv
        #     sub_m = sub_m * (g_ms * l_ms) + x_ms
        #     sub_m = sub_m * (l_mw * g_mw) + x_mw
        #     sub_m = sub_m * (g_mb * l_mb) + x_mb
        #     sub_m = sub_m * n_ml + x_ml
        #     sub_m = sub_m * n_mc + x_mc
        #     sub_m = sub_m * 4
        #     sub_m_index[ic] = sub_m
        '''
        above for loop is simple for loop
        '''
        # for ic in range(ttm.c_m0()):
        #     sub_m = 0
        #     nic = ic            # when this value become zero, means we no need to go further dimension
        #     if nic != 0:
        #         x_mc = nic % n_mc; nic = nic // n_mc
        #         if nic != 0:
        #             x_ml = nic % n_ml; nic = nic // n_ml
        #             if nic != 0:
        #                 x_mb = nic % l_mb; nic = nic // l_mb
        #                 if nic != 0:
        #                     x_mw = nic % l_mw; nic = nic // l_mw
        #                     if nic != 0:
        #                         x_ms = nic % l_ms; nic = nic // l_ms
        #                         if nic != 0:
        #                             x_mv = nic % n_mv; nic = nic // n_mv
        #                             x_mr = nic % l_mr
        #                             # print("    +-> x_mc:{}, x_ml:{}, x_mb:{}, x_mw:{}, x_ms:{}, x_mv:{}, x_mr:{}".format(x_mc, x_ml, x_mb, x_mw, x_ms, x_mv, x_mr))
        #                             sub_m = x_mr * n_mv + x_mv
        #                         sub_m = sub_m * (g_ms * l_ms) + x_ms
        #                     sub_m = sub_m * (l_mw * g_mw) + x_mw
        #                 sub_m = sub_m * (g_mb * l_mb) + x_mb
        #             sub_m = sub_m * n_ml + x_ml
        #         sub_m = sub_m * n_mc + x_mc
        #         sub_m = sub_m * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M
        #     sub_m_index[ic] = sub_m
        '''
        below is instruction-save approach
        '''
        for ic in range(ttm.c_m0()):
            c_m0 = ttm.c_m0()
            nic = ic
            nd_list = list()
            nd_stride = [n_mc, n_ml, g_mb * l_mb, l_mw * g_mw, g_ms * l_ms, n_mv, 1 ]
            if n_mc != 1 and c_m0 != 1: x_mc = nic % n_mc; nic = nic // n_mc; c_m0 = c_m0 // n_mc; nd_list.insert(0, ('x_mc', x_mc, 1))
            if n_ml != 1 and c_m0 != 1: x_ml = nic % n_ml; nic = nic // n_ml; c_m0 = c_m0 // n_ml; nd_list.insert(0, ('x_ml', x_ml, flatten(nd_stride[0:1])))
            if l_mb != 1 and c_m0 != 1: x_mb = nic % l_mb; nic = nic // l_mb; c_m0 = c_m0 // l_mb; nd_list.insert(0, ('x_mb', x_mb, flatten(nd_stride[0:2])))
            if l_mw != 1 and c_m0 != 1: x_mw = nic % l_mw; nic = nic // l_mw; c_m0 = c_m0 // l_mw; nd_list.insert(0, ('x_mw', x_mw, flatten(nd_stride[0:3])))
            if l_ms != 1 and c_m0 != 1: x_ms = nic % l_ms; nic = nic // l_ms; c_m0 = c_m0 // l_ms; nd_list.insert(0, ('x_ms', x_ms, flatten(nd_stride[0:4])))
            if n_mv != 1 and c_m0 != 1: x_mv = nic % n_mv; nic = nic // n_mv; c_m0 = c_m0 // n_mv; nd_list.insert(0, ('x_mv', x_mv, flatten(nd_stride[0:5])))
            if l_mr != 1 and c_m0 != 1: x_mr = nic % l_mr;                    c_m0 = c_m0 // l_mr; nd_list.insert(0, ('x_mr', x_mr, flatten(nd_stride[0:6])))
            sub_m = 0
            # print(f"ic:{ic}, nd_list:{nd_list}")

            c_m0 = ttm.c_m0()
            if n_mc != 1 and c_m0 != 1: sub_m = (x_mc * 1                       ) | sub_m; c_m0 = c_m0 // n_mc
            if n_ml != 1 and c_m0 != 1: sub_m = (x_ml * flatten(nd_stride[0:1]) ) | sub_m; c_m0 = c_m0 // n_ml
            if l_mb != 1 and c_m0 != 1: sub_m = (x_mb * flatten(nd_stride[0:2]) ) | sub_m; c_m0 = c_m0 // l_mb
            if l_mw != 1 and c_m0 != 1: sub_m = (x_mw * flatten(nd_stride[0:3]) ) | sub_m; c_m0 = c_m0 // l_mw
            if l_ms != 1 and c_m0 != 1: sub_m = (x_ms * flatten(nd_stride[0:4]) ) | sub_m; c_m0 = c_m0 // l_ms
            if n_mv != 1 and c_m0 != 1: sub_m = (x_mv * flatten(nd_stride[0:5]) ) | sub_m; c_m0 = c_m0 // n_mv
            if l_mr != 1 and c_m0 != 1: sub_m = (x_mr * flatten(nd_stride[0:6]) ) | sub_m; c_m0 = c_m0 // l_mr

            # for i in range(len(nd_list)):
            #     i_id = nd_list[i][1]
            #     i_st = nd_list[i][2]
            #     sub_m += i_id * i_st

            sub_m = sub_m * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M
            sub_m_index[ic] = sub_m
        return sub_m_index


    def get_m_index_from_m1_m0(self, m_idx):
        assert len(self.gemm_m_m0_m1) == 2
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_idx
        n_m0, n_m1 = self.gemm_m_m0_m1[0], self.gemm_m_m0_m1[1]
        i_m0 = m_idx % n_m0
        i_m1 = m_idx // n_m0
        return i_m0 * n_m1 + i_m1

    def get_m_index_per_group_m1_m0(self):
        m_index_per_group = copy.copy(self.get_m_index_per_group())
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_index_per_group
        assert len(self.gemm_m_m0_m1) == 2
        assert len(m_index_per_group) == self.coalescing_groups
        for ig in range(len(m_index_per_group)):
            for i_cm0 in range(len(m_index_per_group[ig])):
                for i_t in range(len(m_index_per_group[ig][i_cm0])):
                    m_index = m_index_per_group[ig][i_cm0][i_t]
                    m_index_per_group[ig][i_cm0][i_t] = self.get_m_index_from_m1_m0(m_index)
        return m_index_per_group

class igemm_coalescing_store_xdlops_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_xdlops_t
        self.ctrl = ctrl

    def name(self):
        return ''

    def init_co_lds_offset(self, v_co_sst, v_co_sld, v_gemm_im, v_gemm_in, v_tid, v_tmp4):
        ctrl = self.ctrl
        g_mr, g_ms, g_mw, g_mb, g_mt = ctrl.get_subgroups()
        l_mr, l_ms, l_mw, l_mb, l_mt = ctrl.get_subgroup_length()

        with self._deferred_context():
            self._emit(f"; init_co_lds_offset for xdlops")
            '''
            gemm_m_shrink is in multi-dimension.
            then, consider that introduced by granularity
            '''
            self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {igemm_log2(ctrl.cxm.lanegroup_m_per_thread())}, v[{v_gemm_im}]")
            self._emit(f"v_and_b32 v[{v_tmp4}],  {ctrl.cxm.lanegroup_m_per_cluster() - 1} v[{v_tmp4}]   ; thread id of lanegroup_m_per_cluster")
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {igemm_log2(ctrl.cxm.lanegroup_m_per_thread())}, v[{v_tmp4}]")

            if ctrl.cxm.block_m_per_lanegroup() != 1:
                length_above_block_m_per_lanegroup = ctrl.cxm.lanegroup_m_per_block() * ctrl.cxm.lanegroup_m_per_cluster() * \
                                                    ctrl.cxm.lanegroup_m_per_thread()
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}+1], {igemm_log2(length_above_block_m_per_lanegroup)}, v[{v_gemm_im}]")
                self._emit(f"v_and_b32 v[{v_tmp4}+1], {ctrl.cxm.block_m_per_lanegroup() - 1}  , v[{v_tmp4}+1]   ; thread id of block_m_per_lanegroup")
                assert length_above_block_m_per_lanegroup % g_mb == 0, f"length_above_block_m_per_lanegroup:{length_above_block_m_per_lanegroup}, g_mb:{g_mb}"
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}+1], {igemm_log2(length_above_block_m_per_lanegroup // g_mb)}, v[{v_co_sst}]")

            if ctrl.cxm.waves_per_m() != 1:
                length_above_waves_per_m = ctrl.cxm.wave_step_m * ctrl.cxm.lanegroup_m_per_wave() * \
                                                    ctrl.cxm.lanegroup_m_per_block() * ctrl.cxm.block_m_per_lanegroup() * \
                                                    ctrl.cxm.lanegroup_m_per_thread() * ctrl.cxm.lanegroup_m_per_cluster()
                self._emit(f"v_lshrrev_b32 v[{v_tmp4}+2], {igemm_log2(length_above_waves_per_m)}, v[{v_gemm_im}]  ; thread id of waves_per_m")
                assert length_above_waves_per_m % (g_ms * g_mw * g_mb) == 0, f"length_above_waves_per_m:{length_above_waves_per_m}, g_ms:{g_ms} g_mw:{g_mw} g_mb:{g_mb}"
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}+2], {igemm_log2(length_above_waves_per_m // (g_ms * g_mw * g_mb))}, v[{v_co_sst}]")

            self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {igemm_log2(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_co_sst}]")
            self._emit(f"v_lshlrev_b32 v[{v_tmp4}+1], {igemm_log2(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_gemm_in}]   ; implicit transpose with m granularity:{AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M} while store")
            self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}], {igemm_log2(ctrl.cxm.macro_tile_n * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_tmp4}+1]")

            # gemm_m_shrink = g_mw * g_mb * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M     # => granularity shrink
            # gemm_m_shrink = AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M
            # if gemm_m_shrink != 1:
            #     self._emit(f"v_lshrrev_b32 v[{v_tmp4}], {igemm_log2(gemm_m_shrink)}, v[{v_gemm_im}]")
            #     self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp4}], {igemm_log2(ctrl.cxm.macro_tile_n * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_tmp4}+1]")
            # else:
            #     assert False, "impossible"
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {igemm_log2(ctrl.data_byte)}, v[{v_co_sst}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sld}], {igemm_log2(ctrl.data_byte * ctrl.vector_write_out * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_tid}]")

        return self._get_deferred()

    def init_co_sub_m_index(self, v_co_sub_m_index, v_tid, v_tmp6):
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        ctrl = self.ctrl
        # need use v_co_sub_m_index to calculate v offset in m direction
        g_mr, g_ms, g_mw, g_mb, g_mt = ctrl.get_subgroups()
        l_mr, l_ms, l_mw, l_mb, l_mt = ctrl.get_subgroup_length()
        n_mc = ctrl.cxm.lanegroup_m_per_cluster()       # this iteration is among different thread
        n_ml = ctrl.cxm.block_m_per_lanegroup()         # this iteration is among different thread
        n_mv = ctrl.cxm.waves_per_m()                   # this iteration is among different thread
        ttm = ctrl.get_transposed_thread_mapping()
        nd_stride = [n_mc, n_ml, g_mb * l_mb, l_mw * g_mw, g_ms * l_ms, n_mv, 1 ]

        with self._deferred_context():
            self._emit(f"; init_co_sub_m_index xdlops, block_size:{ctrl.cxm.block_size()}, macro-tile:{ctrl.cxm.macro_tile_m}x{ctrl.cxm.macro_tile_n} sub_m_index:{ctrl.get_co_sub_m_index()}")
            self._emit(f"; g_mr:{g_mr}, g_ms:{g_ms}, g_mw:{g_mw}, g_mb:{g_mb}, g_mt:{g_mt} | l_mr:{l_mr}, l_ms:{l_ms}, l_mw:{l_mw}, l_mb:{l_mb}, l_mt:{l_mt} | n_mc:{n_mc}, n_ml:{n_ml}, n_mv:{n_mv}")
            self._emit(f"; nd_stride:{nd_stride}")
            c_m0 = ttm.c_m0()
            if c_m0 == 1:
                # give a chance to early exit, only let the co_sub_m to be zero
                self._emit(f"v_mov_b32 v[{v_co_sub_m_index}], 0")
            else:
                if ctrl.vector_write_out == 1:
                    self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(ctrl.cxm.macro_tile_n)}, v[{v_tid}]   ; get tid along m")
                else:
                    self._emit(f"v_lshlrev_b32 v[{v_tmp6}], {igemm_log2(ctrl.vector_write_out)}, v[{v_tid}]")
                    self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(ctrl.cxm.macro_tile_n)}, v[{v_tmp6}]  ; get tid along m")

                v_idx = 0
                # iterate all dimensions
                if n_mc != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {n_mc - 1}, v[{v_co_sub_m_index}]                   ; => x_mc")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // n_mc
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(n_mc)}  ,v[{v_co_sub_m_index}]")
                if n_ml != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {n_ml - 1}, v[{v_co_sub_m_index}]                   ; => x_ml")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // n_ml
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(n_ml)}  ,v[{v_co_sub_m_index}]")
                if l_mb != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_mb - 1}, v[{v_co_sub_m_index}]                   ; => x_mb")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // l_mb
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(l_mb)}  ,v[{v_co_sub_m_index}]")
                if l_mw != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_mw - 1}, v[{v_co_sub_m_index}]                   ; => x_mw")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // l_mw
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(l_mw)}  ,v[{v_co_sub_m_index}]")
                if l_ms != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_ms - 1}, v[{v_co_sub_m_index}]                   ; => x_ms")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // l_ms
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(l_ms)}  ,v[{v_co_sub_m_index}]")
                if n_mv != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {n_mv - 1}, v[{v_co_sub_m_index}]                   ; => x_mv")
                    v_idx = v_idx + 1
                    c_m0  = c_m0 // n_mv
                    if c_m0 not in (0, 1):
                        self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(n_mv)}  ,v[{v_co_sub_m_index}]")
                if l_mr != 1 and c_m0 not in (0, 1):
                    self._emit(f"v_and_b32 v[{v_tmp6}+{v_idx}], {l_mr - 1}, v[{v_co_sub_m_index}]                   ; => x_mr")

                # indeed accoroding to current implementation, at most 2 tmp vgpr is used.
                # we keep 6 temp here in case in the future there might be different mapping
                assert v_idx <= 5, "since current we only assign 6 vgpr to do this nd split-merge, so larger than 6 vgpr currently not supported"

                class pretty_accumulate_co_sub_m_t(object):
                    def __init__(self):
                        self.first = 1
                    def __call__(self, v_co_sub_m_index, v_idx, k_multiplier):
                        '''
                        return a string to calculate v_co_sub_m_index = v_idx * k_multiplier + v_co_sub_m_index
                        '''
                        assert igemm_is_pow2(k_multiplier)
                        if self.first == 1:
                            self.first = 0
                            if k_multiplier == 1:
                                return f"v_mov_b32 v[{v_co_sub_m_index}], v[{v_idx}]"
                            else:
                                return f"v_lshlrev_b32 v[{v_co_sub_m_index}], {igemm_log2(k_multiplier)}, v[{v_idx}]"
                        else:
                            if k_multiplier == 1:
                                return  f"v_add_u32 v[{v_co_sub_m_index}], v[{v_idx}], v[{v_co_sub_m_index}]"
                            else:
                                return f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_idx}], {igemm_log2(k_multiplier)}, v[{v_co_sub_m_index}]"

                c_m0 = ttm.c_m0()
                v_idx_r = 0
                # self._emit(f"v_mov_b32 v[{v_co_sub_m_index}], 0")
                # if n_mc != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_add_u32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], v[{v_co_sub_m_index}]  ; => accumulate x_mc")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // n_mc
                # if n_ml != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {igemm_log2(flatten(nd_stride[0:1]))}, v[{v_co_sub_m_index}]  ; => accumulate x_ml")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // n_ml
                # if l_mb != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {igemm_log2(flatten(nd_stride[0:2]))}, v[{v_co_sub_m_index}]  ; => accumulate x_mb")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // l_mb
                # if l_mw != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {igemm_log2(flatten(nd_stride[0:3]))}, v[{v_co_sub_m_index}]  ; => accumulate x_mw")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // l_mw
                # if l_ms != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {igemm_log2(flatten(nd_stride[0:4]))}, v[{v_co_sub_m_index}]  ; => accumulate x_ms")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // l_ms
                # if n_mv != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {igemm_log2(flatten(nd_stride[0:5]))}, v[{v_co_sub_m_index}]  ; => accumulate x_mv")
                #     v_idx_r = v_idx_r + 1
                #     c_m0    = c_m0 // n_mv
                # if l_mr != 1 and c_m0 not in (0, 1):
                #     self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}], v[{v_tmp6}+{v_idx_r}], {igemm_log2(flatten(nd_stride[0:6]))}, v[{v_co_sub_m_index}]  ; => accumulate x_mr")
                accumulate_co_sub_m = pretty_accumulate_co_sub_m_t()
                if n_mc != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", 1) + "      ; => accumulate x_mc")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // n_mc
                if n_ml != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:1])) + "      ; => accumulate x_ml")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // n_ml
                if l_mb != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:2])) + "      ; => accumulate x_mb")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // l_mb
                if l_mw != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:3])) + "      ; => accumulate x_mw")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // l_mw
                if l_ms != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:4])) + "      ; => accumulate x_ms")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // l_ms
                if n_mv != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:5])) + "      ; => accumulate x_mv")
                    v_idx_r = v_idx_r + 1
                    c_m0    = c_m0 // n_mv
                if l_mr != 1 and c_m0 not in (0, 1):
                    self._emit(accumulate_co_sub_m(v_co_sub_m_index, f"{v_tmp6}+{v_idx_r}", flatten(nd_stride[0:6])) + "      ; => accumulate x_mr")

                self._emit(f"v_lshlrev_b32 v[{v_co_sub_m_index}], {igemm_log2(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)}, v[{v_co_sub_m_index}]")

                assert v_idx == v_idx_r, "please check!"

        return self._get_deferred()

    def init_co_sub_n_index(self, v_co_sub_n_index, v_tid, v_tmp2):
        '''
        in n dimension, always have one thread per column
        '''
        ctrl = self.ctrl
        # need use v_co_sub_n_index to calculate v offset in n direction

        with self._deferred_context():
            self._emit(f"; init_co_sub_n_index xdlops")
            if ctrl.vector_write_out == 1:
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.cxm.macro_tile_n - 1}, v[{v_tid}]")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {igemm_log2(ctrl.vector_write_out)}, v[{v_tid}]")
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.cxm.macro_tile_n - 1}, v[{v_tmp2}]")
        return self._get_deferred()

    def __call__(self, a_c, v_c, v_co_sst, v_co_sld, s_p_out, v_out_offset, s_out_offset, s_gemm_m0_stride, s_gemm_m1_stride, s_tmp6, v_store_flag = None, s_k = None, v_cur_k = None, s_block_gtc_ik = None, v_co_sub_m_index = None, v_tmp0 = None):

        # if no need s_out_offset, set to integer 0
        # if no need flag to dicide store, set v_store_flag to 0
        def flatten(x):
            from functools import reduce
            return reduce(lambda a, b: a*b, x, 1)
        ctrl = self.ctrl
        a_c = sym_t(a_c)
        v_c = sym_t(v_c)
        v_co_sst = sym_t(v_co_sst)
        v_co_sld = sym_t(v_co_sld)
        s_tmp6 = sym_t(s_tmp6)

        if s_k is not None:
            s_k = sym_t(s_k)
            v_cur_k = sym_t(v_cur_k)
            s_block_gtc_ik = sym_t(s_block_gtc_ik)
            v_co_sub_m_index = sym_t(v_co_sub_m_index)
            v_tmp0 = sym_t(v_tmp0)

        g_mr, g_ms, g_mw, g_mb, g_mt = ctrl.get_subgroups()
        l_mr, l_ms, l_mw, l_mb, l_mt = ctrl.get_subgroup_length()
        n_mc = ctrl.cxm.lanegroup_m_per_cluster()       # this iteration is among different thread
        n_ml = ctrl.cxm.block_m_per_lanegroup()         # this iteration is among different thread
        n_mv = ctrl.cxm.waves_per_m()                   # this iteration is among different thread

        n_nc = ctrl.cxm.lanegroup_n_per_cluster()       # this iteration is among different thread
        n_nl = ctrl.cxm.block_n_per_lanegroup()         # this iteration is among different thread
        n_nv = ctrl.cxm.waves_per_n()                   # this iteration is among different thread


        ttm = ctrl.get_transposed_thread_mapping()
        nd_stride = [n_mc, n_ml, g_mb * l_mb, l_mw * g_mw, g_ms * l_ms, n_mv, 1 ]

        no_s_out_offset = s_out_offset is None

        # for xdlops, always consider granularity in column, hence here is always ds_write_b128/ds_read_b128
        inst_sst = inst_ds_write_t(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M * ctrl.data_byte)
        inst_sld = inst_ds_read_t(AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M * ctrl.data_byte)
        if ctrl.gemm_k_global_split: 
            inst_gst = inst_buffer_atomic_add_dword_t(ctrl.vector_write_out) 
        else:
            inst_gst = inst_buffer_store_dword_t(ctrl.vector_write_out)
       

        s_out_offset_itr = sym_t(s_tmp6(0))
        # s_thread_m_stride = sym_t(s_tmp4(1))

        if ctrl.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            m_index_per_group = ctrl.get_m_index_per_group()
        else:
            m_index_per_group = ctrl.get_m_index_per_group_m1_m0()
        # thread_m_stride = ctrl.get_thread_m_stride()

        assert len(m_index_per_group) == ctrl.coalescing_groups

        t_mr, t_nr, t_ms, t_ns, t_mw, t_nw, t_mb, t_mt = ctrl.cxm.acc_c_lengths()
        s_mr, s_nr, s_ms, s_ns, s_mw, s_nw, s_mb, s_mt = t_nr * t_ms * t_ns * t_mw * t_nw * t_mb * t_mt, \
                                                            t_ms * t_ns * t_mw * t_nw * t_mb * t_mt, \
                                                            t_ns * t_mw * t_nw * t_mb * t_mt, \
                                                            t_mw * t_nw * t_mb * t_mt, \
                                                            t_nw * t_mb * t_mt, \
                                                            t_mb * t_mt, \
                                                            t_mt, \
                                                            1

        i_g_list = list()
        for i_g_mr, i_g_ms, i_g_mw, i_g_mb, i_g_mt in itertools.product(range(g_mr), range(g_ms), range(g_mw), range(g_mb), range(g_mt)):
            i_g_list.append((i_g_mr, i_g_ms, i_g_mw, i_g_mb, i_g_mt))

        agpr_consume_list = list()      # used for assertion
        lds_size_per_group = (ctrl.cxm.macro_tile_m * ctrl.cxm.macro_tile_n // ctrl.coalescing_groups) * ctrl.data_byte # used for assertion

        with self._deferred_context():
            self._emit(f"; coalescing store, mapping:{ctrl.cxm.serialize()}")
            self._emit(f"; coalescing_groups:{ctrl.coalescing_groups}, num_dword_per_group:{ctrl.get_num_dword_per_group()}")
            self._emit(f"; init_co_sub_m_index xdlops, block_size:{ctrl.cxm.block_size()}, macro-tile:{ctrl.cxm.macro_tile_m}x{ctrl.cxm.macro_tile_n} sub_m_index:{ctrl.get_co_sub_m_index()}")
            self._emit(f"; g_mr:{g_mr}, g_ms:{g_ms}, g_mw:{g_mw}, g_mb:{g_mb}, g_mt:{g_mt} | l_mr:{l_mr}, l_ms:{l_ms}, l_mw:{l_mw}, l_mb:{l_mb}, l_mt:{l_mt} | n_mc:{n_mc}, n_ml:{n_ml}, n_mv:{n_mv}")
            self._emit(f"; nd_stride:{nd_stride}")

            # some cases that 
            if ctrl.can_skip_coalescing():
                self._emit(f";[X] coalescing through LDS is skipped")

            for i_group in range(ctrl.coalescing_groups):
                m_index_start_per_group = m_index_per_group[i_group][0][0]
                m0_index_start_per_group, m1_index_start_per_group = ctrl.get_m0_m1_index(m_index_start_per_group)

                i_g_mr, i_g_ms, i_g_mw, i_g_mb, i_g_mt = i_g_list[i_group]
                a_group_start_index = i_g_mr * l_mr * s_mr + i_g_ms * l_ms * s_ms + i_g_mw * l_mw * s_mw + i_g_mb * l_mb * s_mb + i_g_mt * l_mt * s_mt

                self._emit(f"; start group {i_group}, i_g_mr:{i_g_mr}, i_g_ms:{i_g_ms}, i_g_mw:{i_g_mw}, i_g_mb:{i_g_mb}, i_g_mt:{i_g_mt}, m index start from {m_index_start_per_group}")
                if not ctrl.can_skip_coalescing():
                    self._emit(f"s_barrier")

                vgpr_index_acc = 0
                for i_mr, i_ms, i_mw, i_mb in itertools.product(range(l_mr), range(l_ms), range(l_mw), range(l_mb)):
                    gpr_m_offset = i_mr * s_mr + i_ms * s_ms + i_mw * s_mw + i_mb * s_mb
                    sst_m_offset = i_mr *  n_mv * l_ms * l_mw * l_mb * n_ml * n_mc + \
                                    i_ms * l_mw * l_mb * n_ml * n_mc + \
                                    i_mw * l_mb * n_ml * n_mc + \
                                    i_mb * n_ml * n_mc              # ATTENTION! here is not multiplied by granularity_m, since it is shrinked to n direction, when everything is in LDS
                    # iterate through all m within current group
                    for i_nr, i_ns, i_nw in itertools.product(range(t_nr), range(t_ns), range(t_nw)):
                        gpr_n_offset = i_nr * s_nr + i_ns * s_ns + i_nw * s_nw
                        sst_n_offset = i_nr * n_nv * ctrl.cxm.wave_step_n * ctrl.cxm.lanegroup_n_per_wave() * n_nl * n_nc + \
                                     i_ns * ctrl.cxm.lanegroup_n_per_wave() * n_nl * n_nc + \
                                     i_nw * n_nl * n_nc
                        # self._emit(f" => ctrl.cxm.wave_step_n:{ctrl.cxm.wave_step_n}, ctrl.cxm.lanegroup_n_per_wave():{ctrl.cxm.lanegroup_n_per_wave()}, n_nl:{n_nl}, n_nc:{n_nc}")
                        agpr_index = a_group_start_index + gpr_m_offset + gpr_n_offset
                        #vgpr_index = gpr_m_offset + gpr_n_offset
                        vgpr_index = vgpr_index_acc
                        sst_offset = (sst_m_offset * ctrl.cxm.macro_tile_n * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M + \
                                     sst_n_offset * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M) * ctrl.data_byte
                        assert sst_offset < lds_size_per_group and sst_offset + (m_index_per_group[i_group][-1][0] -  m_index_per_group[i_group][0][0]) * ctrl.data_byte < lds_size_per_group

                        self._emit(f"v_accvgpr_read_b32 v[{v_c(vgpr_index + 0)}], a[{a_c(agpr_index + 0)}]") ; agpr_consume_list.append(agpr_index + 0)
                        self._emit(f"v_accvgpr_read_b32 v[{v_c(vgpr_index + 1)}], a[{a_c(agpr_index + 1)}]") ; agpr_consume_list.append(agpr_index + 1)
                        self._emit(f"v_accvgpr_read_b32 v[{v_c(vgpr_index + 2)}], a[{a_c(agpr_index + 2)}]") ; agpr_consume_list.append(agpr_index + 2)
                        self._emit(f"v_accvgpr_read_b32 v[{v_c(vgpr_index + 3)}], a[{a_c(agpr_index + 3)}]") ; agpr_consume_list.append(agpr_index + 3)
                        # self._emit(f"s_nop 4")    # might consider nop to solve RAW
                        if not ctrl.can_skip_coalescing():
                            idword = sst_offset // (ctrl.data_byte * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M)
                            self._emit(inst_sst(v_co_sst(), v_c(vgpr_index), sst_offset) + \
                                f"   ; idword:{idword}({idword // ctrl.cxm.macro_tile_n},{idword % ctrl.cxm.macro_tile_n}),  {sst_m_offset}x{sst_n_offset} |" + \
                                f" /{AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M}, i_mr:{i_mr}, i_ms:{i_ms}, i_mw:{i_mw}, i_mb:{i_mb}  x  i_nr:{i_nr}, i_ns:{i_ns}, i_nw:{i_nw}")
                        vgpr_index_acc += AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M

                issue_list = []
                if not ctrl.can_skip_coalescing():
                    self._emit(f"s_waitcnt lgkmcnt(0)")
                    self._emit(f"s_barrier")
                    self._emit(f";   load from lds")
                    #for i_mr, i_ms, i_mw, i_mb in itertools.product(range(l_mr), range(l_ms), range(l_mw), range(l_mb)):
                    for i_d in range(ctrl.get_num_dword_per_group() // AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M):
                        vgpr_index = i_d * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M
                        sld_offset = i_d * AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M * ctrl.block_size * ctrl.vector_write_out * ctrl.data_byte
                        self._emit(inst_sld(v_c(vgpr_index), v_co_sld(), sld_offset))
                        issue_list.append(inst_sld.get_issues(sld_offset))

                if v_store_flag is not None and type(v_store_flag) is str:
                    self._emit(f"v_cmpx_eq_u32 vcc, 1, v[{v_store_flag}]")

                self._emit(f";   store to global, m index start from {m_index_start_per_group}, m0:{m0_index_start_per_group}, m1:{m1_index_start_per_group}")

                def emit_calculate_s_out_offset_itr(i_m, i_m0, i_m1):
                    # self._emit(f"; i_m:{i_m},  i_m0:{i_m0}xi_m1:{i_m1}")
                    comments = f"   ; i_m:{i_m}(i_m0:{i_m0},i_m1:{i_m1})"
                    if s_gemm_m0_stride is not None:
                        self._emit(f"s_mul_i32 s[{s_tmp6(2)}], {i_m0}, s[{s_gemm_m0_stride}]")
                        self._emit(f"s_mul_i32 s[{s_tmp6(3)}], {i_m1}, s[{s_gemm_m1_stride}]")
                        self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp6(2)}], s[{s_tmp6(3)}]" + comments)
                        if not no_s_out_offset:
                            self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_out_offset}], s[{s_out_offset_itr()}] ")
                    else:
                        '''
                        no m0_stride, which indicate m0, m1 is continuous, no need to deal with m0, m1 seperately
                        '''
                        if i_m == 0:
                            if no_s_out_offset:
                                self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], 0" + comments)
                                if s_k is not None:
                                    self._emit(f"v_add_u32 v[{v_cur_k()}], s[{s_block_gtc_ik()}], v[{v_co_sub_m_index()}]")
                                    self._emit(f"v_mov_b32 v[{v_tmp0()}], v[{v_cur_k()}]")
                            else:
                                self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_out_offset}]" + comments)
                        elif i_m == 1:
                            if no_s_out_offset:
                                self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}]" + comments)
                                if s_k is not None:
                                    self._emit(f"v_add_u32 v[{v_tmp0()}], 1, v[{v_cur_k()}]")
                            else:
                                self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}], s[{s_out_offset}]" + comments)
                        else:
                            if no_s_out_offset:
                                self._emit(f"s_mul_i32 s[{s_out_offset_itr()}], {i_m}, s[{s_gemm_m1_stride}]" + comments)
                                if s_k is not None:
                                    self._emit(f"v_add_u32 v[{v_tmp0()}], {i_m}, v[{v_cur_k()}]")
                            else:
                                self._emit(f"s_mul_i32 s[{s_tmp6(3)}], {i_m}, s[{s_gemm_m1_stride}]")
                                self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp6(3)}], s[{s_out_offset}]" + comments)

                emit_calculate_s_out_offset_itr(m_index_start_per_group, m0_index_start_per_group, m1_index_start_per_group)
                i_m0_start, i_m1_start =  m0_index_start_per_group, m1_index_start_per_group
                for i_gst in range(ctrl.get_num_dword_per_group() // ctrl.vector_write_out):
                    if len(issue_list) != 0:
                        if i_gst % AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M == 0:
                            i_issues =  (i_gst // AMDGPU_XDLOPS_LANEGROUP_GRANULARITY_M) + 1
                            i_issue_list = issue_list[i_issues:]
                            i_issue_cnt = igemm_flatten_list_accumulate(i_issue_list) if len(i_issue_list) != 0 else 0
                            self._emit(f"s_waitcnt lgkmcnt({i_issue_cnt})")
                    # vdata, vaddr, srsrc, soffset, offset
                    if s_k is not None:
                        self._emit(f"v_cmp_gt_u32 vcc, s[{s_k()}], v[{v_tmp0()}]")
                        self._emit(f"s_and_saveexec_b64 s[{s_tmp6(4)}:{s_tmp6(5)}], vcc")
                    self._emit(inst_gst(v_c(i_gst*ctrl.vector_write_out), v_out_offset, s_p_out, s_out_offset_itr(), 0))
                    if s_k is not None:
                        self._emit(f"s_or_b64 exec, exec, s[{s_tmp6(4)}:{s_tmp6(5)}]")
                    if i_gst != (ctrl.get_num_dword_per_group() // ctrl.vector_write_out) - 1:
                        i_m = m_index_per_group[i_group][0][i_gst+1]
                        # self._emit(f"; >>>>>> i_m :{i_m}, i_gst:{i_gst}, m_index_per_group[i_group][0]:{m_index_per_group[i_group][0]}")
                        i_m0, i_m1 = ctrl.get_m0_m1_index(i_m)
                        emit_calculate_s_out_offset_itr(i_m, i_m0, i_m1)
                if v_store_flag is not None and type(v_store_flag) is str:
                    self._emit(f"s_mov_b64 exec, -1")

            # do some assert
            agpr_consume_list.sort()
            assert agpr_consume_list == [x for x in range(ctrl.cxm.total_acc_c())], f"agpr_consume_list:{agpr_consume_list}"
            #if agpr_consume_list != [x for x in range(ctrl.cxm.total_acc_c())]:
            #    self._emit(f"; [WRONG!] agpr ~~~~~~~~, {agpr_consume_list}")
        return self._get_deferred()
