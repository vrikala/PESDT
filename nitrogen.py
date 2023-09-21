####
# Unused Nitrogen related functions, which still depend on idl
# Cut from process.py line 1288
# NON-FUNCTIONAL AS IS

        ###############################################################
        # N II AFG feature (399 < wv < 410 nm)
        ###############################################################
        if self.calc_NII_afg_feature and self.diag == 'KT3':
            if self.imp1_atom_num or self.imp2_atom_num:

                print('Calculating NII ADAS AFG feature ', self.chord_num)

                # ionization balance flag
                ionbal=0
                afg_adasn1_tmp = []
                IDL_afg_adasn1= idl.export_function("calc_nii_emiss__afg_adasn1")
                for dl_idx, dl_val in enumerate(self.los_1d['l']):
                    if self.imp1_atom_num == 7:
                        nii_cm3 = 1.e-06 * self.los_1d['imp1_den'][dl_idx][1]
                        niii_cm3 = 1.e-06 * self.los_1d['imp1_den'][dl_idx][2]
                    elif self.imp2_atom_num == 7:
                        nii_cm3 = 1.e-06 * self.los_1d['imp2_den'][dl_idx][1]
                        niii_cm3 = 1.e-06 * self.los_1d['imp2_den'][dl_idx][2]
                    # call IDL calc_nii_emiss__afg_adas1 function (returns: wv[nm] intensity [ph s-1 cm-3])
                    # if ionbal=1, nii_cm3 and niii_cm3 are ignored
                    res_dict = IDL_afg_adasn1(self.los_1d['te'][dl_idx], 1.e-06 * self.los_1d['ne'][dl_idx], nii_cm3, niii_cm3, ionbal)
                    if dl_idx == 0: wv_nm = res_dict['wv'] / 10.
                    if ionbal == 0:
                        # convert to radiance: ph s-1 m-2 sr-1
                        afg_adasn1_tmp.append( (1./ (4.*np.pi)) * 1.e06*res_dict['intensity'] * self.los_1d['ortho_delL'][dl_idx] )
                    else:
                        afg_adasn1_tmp.append( res_dict['intensity'] * self.los_1d['ortho_delL'][dl_idx] )

                afg_adasn1_arr = np.asarray(afg_adasn1_tmp)

                # store spectra
                if ionbal == 0:
                    units = 'nm, ph s^-1 m^-2 sr^-1'
                else:
                    units = 'arb. units'
                self.los_1d_spectra['afg_adasn1'] = {'wave':wv_nm, 'intensity':afg_adasn1_arr, 'units':units}
                self.los_int_spectra['afg_adasn1'] = {'wave':wv_nm, 'intensity':np.sum(afg_adasn1_arr, axis=0), 'units':units}
                # # convert numpy array to list for JSON serialization
                self.los_1d_spectra['afg_adasn1']['wave'] = self.los_1d_spectra['afg_adasn1']['wave'].tolist()
                self.los_1d_spectra['afg_adasn1']['intensity'] = self.los_1d_spectra['afg_adasn1']['intensity'].tolist()
                self.los_int_spectra['afg_adasn1']['wave'] = self.los_int_spectra['afg_adasn1']['wave'].tolist()
                self.los_int_spectra['afg_adasn1']['intensity'] = self.los_int_spectra['afg_adasn1']['intensity'].tolist()

                # DEBUGGING
                # fig, ax = plt.subplots(ncols=1)
                # ax.semilogy(self.los_int_spectra['afg_adasn1']['wave'], self.los_int_spectra['afg_adasn1']['intensity'], 'ok', lw=3.0)
                # for dl_idx, dl_val in enumerate(self.los_1d['l']):
                #     ax.semilogy(self.los_int_spectra['afg_adasn1']['wave'], self.los_1d_spectra['afg_adasn1']['intensity'][dl_idx], 'xk')

                # GENERATE SYNTHETIC NII AFG SPECTRA CONVOLVED WITH INST. FN
                KT3B_1200_instfwhm = 0.085 #nm
                KT3B_1200_dwv = 0.0135 #nm
                wv_nm_fine = np.arange(398, 410, KT3B_1200_dwv)
                afg_adasn1_spectrum = np.zeros((len(wv_nm_fine)))
                for iline_int, vline_int in enumerate(self.los_int_spectra['afg_adasn1']['intensity']):
                    # Convert line intensity to Gaussian with FWHM = KT3B_1200_instfwhm [units: phs/s/m2/sr/nm]
                    afg_adasn1_spectrum += gaussian(self.los_int_spectra['afg_adasn1']['wave'][iline_int], wv_nm_fine,
                                                    self.los_int_spectra['afg_adasn1']['intensity'][iline_int], KT3B_1200_instfwhm)
                # NOW ADD 6-2 LINE AND FF-FB CONTINUUM
                # f = interp1d(self.los_int_spectra['stark']['wave'], self.los_int_spectra['stark']['intensity'])
                # stark_interp = f(wv_nm_fine)
                f = interp1d(self.los_int_spectra['ff_fb_continuum']['wave'], self.los_int_spectra['ff_fb_continuum']['intensity'])
                ff_fb_interp = f(wv_nm_fine)

                # DEBUGGING
                # fig, ax = plt.subplots(ncols=1)
                # ax.plot(wv_nm_fine, ff_fb_interp+afg_adasn1_spectrum, '-k', lw=2.0)
                # plt.show()

                # store spectra
                # self.los_int_spectra['afg_adasn1_kt3b1200'] = {'wave':wv_nm_fine, 'intensity':ff_fb_interp+afg_adasn1_spectrum, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                self.los_int_spectra['afg_adasn1_kt3b1200'] = {'wave':wv_nm_fine, 'intensity':afg_adasn1_spectrum, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                # convert numpy array to list for JSON serialization
                self.los_int_spectra['afg_adasn1_kt3b1200']['wave'] = list(self.los_int_spectra['afg_adasn1_kt3b1200']['wave'])
                self.los_int_spectra['afg_adasn1_kt3b1200']['intensity'] = list(self.los_int_spectra['afg_adasn1_kt3b1200']['intensity'])
