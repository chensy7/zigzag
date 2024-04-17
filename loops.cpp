int ix, iy, ox, oy, c, k;
double l1_o [L1_O_DEPTH][L1_O_WIDTH]; // width = 32*16b
double dram_o [DRAM_O_DEPTH][DRAM_O_WIDTH]; // does not take into account DRAM BW < DRAM_O_WIDTH
double rf_o [K_S]; // 32*16b

double l1_w [L1_W_DEPTH][L1_W_WIDTH];
double l1_i [L1_I_DEPTH][L1_I_WIDTH];

TODO: DRAM for weights and inputs

for (int ox_t = 0; ox_t < OX_T; ox_t++) {
    int dram_o_addr = ox_t;
    l1_o[0:K_T*OY_T] = dram[dram_o_addr];
    int dram_i_addr = ix_t;
    l1_o[0:K_T*OY_T] = dram[dram_o_addr];
    for (int fx_t = 0; fx_t < FX_T; fx_t++) {
        for (int fy_t = 0; fy_t < FY_T; fy_t++) {
            for (int k_t = 0; k_t < K_T; k_t++) {
                for (int c_t = 0; c_t < C_T; c_t++) {
                    for (int oy_t = 0; oy_t < OY_T; oy_t++) {
                        ix = fx_t + ox_t * STRIDE;
                        iy = fy_t + oy_t * STRIDE;
                        int l1_i_addr = c_t * IX * IY + ix * IX + iy;
                        double l1_i_r = l1_i[l1_i_addr];
                        int l1_w_addr = fx_t * (FY_T * K_T * C_T) + fy_t * (K_T * C_T) + k_t * C_T + c_t;
                        double l1_w_r = l1_w[l1_w_addr];
                        int l1_o_addr = k_t * OY_T + oy_t;
                        rf_o = l1_o[l1_o_addr];
                        for (int k_s = 0; k_s < K_S; k_s++) {
                            for (int c_s = 0; c_s < C_S; c_s++) {
                                c = c_s + c_t * C_T;
                                rf_o[k_s] += l1_w_r[k_s][c_s] * l1_i_r[c_s];
                            }
                        }
                        l1_o[l1_o_addr] = rf_o;
                    }
                }
            }
        }
    }
    dram_o[dram_o_addr] = l1_o[0:K_T*OY_T];
} 

things to discuss:
1. bw mem < ideal bw; how to coalesce reads from multiple cycles
2. same problem for DRAM
3. flexibly add double buffering?