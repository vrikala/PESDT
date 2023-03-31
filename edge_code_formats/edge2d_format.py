
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import sys
import pickle

from shapely.geometry import Polygon, LineString

from .. import process

import idlbridge as idl

class Edge2D():

    def __init__(self, tranfile, read_fluid_side=True, read_eirene_side=False):

        self.tranfile = tranfile

        self.quad_cells = [] # EDGE2D mesh uses quadralaterals
        self.tri_cells = [] # EIRENE mesh uses triangles
        self.edge2d_dict = {} # Dictionary for storing interesting output other than cell-wise plasma parameters

        if read_fluid_side:
            self.read_edge2d_fluid_side_data_using_idl_eproc()


    def read_edge2d_fluid_side_data_using_idl_eproc(self):

        print('Getting data from ' + self.tranfile )

        IDL_EprocGeom= idl.export_function("EprocGeom")
        self.geom = IDL_EprocGeom(self.tranfile)
        IDL_EprocDataRead= idl.export_function("EprocDataRead")

        self.rmesh = IDL_EprocDataRead(self.tranfile, 'RMESH')
        self.zmesh = IDL_EprocDataRead(self.tranfile, 'ZMESH')
        self.rvertp = IDL_EprocDataRead(self.tranfile, 'RVERTP')
        self.zvertp = IDL_EprocDataRead(self.tranfile, 'ZVERTP')
        self.teve = IDL_EprocDataRead(self.tranfile, 'TEVE')
        self.tev = IDL_EprocDataRead(self.tranfile, 'TEV')
        self.den = IDL_EprocDataRead(self.tranfile, 'DEN')
        self.denel = IDL_EprocDataRead(self.tranfile, 'DENEL')
        self.da = IDL_EprocDataRead(self.tranfile, 'DA')
        if False: # boosting up divertor neutral pressure
            mask = self.zmesh['data'] > 1.4
            self.da['data'][mask] = 30.*self.da['data'][mask]
        self.dm = self.get_eproc_param_temp("EprocDataRead", 'DM')        
        self.korpg = IDL_EprocDataRead(self.tranfile, 'KORPG')
        self.rmesh = IDL_EprocDataRead(self.tranfile, 'RMESH')
        self.sirec = IDL_EprocDataRead(self.tranfile, 'SIREC')
        self.soun = IDL_EprocDataRead(self.tranfile, 'SOUN')

        # GET INNER AND OUTER TARGET DATA
        IDL_EprocRow= idl.export_function("EprocRow") # This function is now depracated!
        self.pflxd_OT = IDL_EprocRow(self.tranfile, 'PFLXD', 'OT', ALL_POINTS=0)
        self.pflxd_IT = IDL_EprocRow(self.tranfile, 'PFLXD', 'IT', ALL_POINTS=0)
        self.teve_OT = IDL_EprocRow(self.tranfile, 'TEVE', 'OT', ALL_POINTS=0)
        self.teve_IT = IDL_EprocRow(self.tranfile, 'TEVE', 'IT', ALL_POINTS=0)
        self.denel_OT = IDL_EprocRow(self.tranfile, 'DENEL', 'OT', ALL_POINTS=0)
        self.denel_IT = IDL_EprocRow(self.tranfile, 'DENEL', 'IT', ALL_POINTS=0)
        self.da_OT = IDL_EprocRow(self.tranfile, 'DA', 'OT', ALL_POINTS=0)
        self.da_IT = IDL_EprocRow(self.tranfile, 'DA', 'IT', ALL_POINTS=0)
        self.dm_OT = IDL_EprocRow(self.tranfile, 'DM', 'OT', ALL_POINTS=0)
        self.dm_IT = IDL_EprocRow(self.tranfile, 'DM', 'IT', ALL_POINTS=0)
        self.psi_OT = IDL_EprocRow(self.tranfile, 'PSI', 'OT', ALL_POINTS=0)
        self.psi_IT = IDL_EprocRow(self.tranfile, 'PSI', 'IT', ALL_POINTS=0)
        # plt.plot(self.teve_OT['xdata'][:self.teve_OT['npts']], self.teve_OT['ydata'][:self.teve_OT['npts']], 'ok')
        # plt.show()

        # GET GEOM INFO
        IDL_EprocGeom= idl.export_function("EprocGeom") # This function is now deprecated!
        self.geom = IDL_EprocGeom(self.tranfile)
        self.geom['zpx']*=-1.0

        # GET MID-PLANE PROFILE
        IDL_EprocRing= idl.export_function("EprocRing")
        self.ne_OMP = IDL_EprocRow(self.tranfile, 'DENEL', 'OMP', ALL_POINTS=0)
        self.te_OMP = IDL_EprocRow(self.tranfile, 'TEVE', 'OMP', ALL_POINTS=0)
        self.ni_OMP = IDL_EprocRow(self.tranfile, 'DEN', 'OMP', ALL_POINTS=0)
        self.ti_OMP = IDL_EprocRow(self.tranfile, 'TEV', 'OMP', ALL_POINTS=0)
        self.da_OMP = IDL_EprocRow(self.tranfile, 'DA', 'OMP', ALL_POINTS=0)
        self.dm_OMP = IDL_EprocRow(self.tranfile, 'DM', 'OMP', ALL_POINTS=0)
        self.psi_OMP = IDL_EprocRow(self.tranfile, 'PSI', 'OMP', ALL_POINTS=0)

        # GET INNER AND OUTER QPARTOT AT Z=-1.2
        # FOR 81472 GEOM THIS CORRESPONDS TO ROWS 66 (IN) AND 26 (OUT)
        # TODO: GENERALIZE TO OTHER GEOMETRIES
        self.qpartot_HFS = IDL_EprocRow(self.tranfile, 'QPARTOT', 66, ALL_POINTS=0)
        self.qpartot_LFS = IDL_EprocRow(self.tranfile, 'QPARTOT', 26, ALL_POINTS=0)
        self.bpol_btot_HFS = IDL_EprocRow(self.tranfile, 'SH', 66, ALL_POINTS=0)
        self.bpol_btot_LFS = IDL_EprocRow(self.tranfile, 'SH', 26, ALL_POINTS=0)
        self.qpartot_HFS_rmesh = IDL_EprocRow(self.tranfile, 'RMESH', 66, ALL_POINTS=0)
        self.qpartot_LFS_rmesh = IDL_EprocRow(self.tranfile, 'RMESH', 26, ALL_POINTS=0)
        self.qpartot_HFS_zmesh = IDL_EprocRow(self.tranfile, 'ZMESH', 66, ALL_POINTS=0)
        self.qpartot_LFS_zmesh = IDL_EprocRow(self.tranfile, 'ZMESH', 26, ALL_POINTS=0)
        # GET POWER CROSSING THE SEPARATRIX
        self.powsol  = IDL_EprocDataRead(self.tranfile, 'POWSOL')

        # GET IMPURITY DATA
        self.imp1_atom_num = None
        self.imp2_atom_num = None
        _imp1_denz = []
        _imp2_denz = []
        _imp1_chrg_idx = []
        _imp2_chrg_idx = []
        self.zch = IDL_EprocDataRead(self.tranfile, 'ZCH') # impurity atomic number (max 2 impurities)
        if self.zch['data'][0] > 0.0:
            self.imp1_atom_num = self.zch['data'][0]
            # First append neutral density, then each charge state density
            _imp1_denz.append(IDL_EprocDataRead(self.tranfile, 'DZ_1'))
            for i in range(int(self.zch['data'][0])):
                if i < 9 :
                    stridx = '0' + str(i+1)
                    _imp1_chrg_idx.append(stridx)
                else:
                    stridx = str(i+1)
                    _imp1_chrg_idx.append(stridx)
                _imp1_denz.append(IDL_EprocDataRead(self.tranfile, 'DENZ'+stridx))
            if self.zch['data'][1] > 0.0:
                self.imp2_atom_num = self.zch['data'][1]
                # First append neutral density, then each charge state density
                _imp2_denz.append(IDL_EprocDataRead(self.tranfile, 'DZ_2'))
                for i in range(int(self.zch['data'][1])):
                    if i + int(self.zch['data'][0]) < 9:
                        stridx = '0' + str(i + 1 + int(self.zch['data'][0]))
                        _imp2_chrg_idx.append(stridx)
                    else:
                        stridx = str(i + 1 + int(self.zch['data'][0]))
                        _imp2_chrg_idx.append(stridx)
                    _imp2_denz.append(IDL_EprocDataRead(self.tranfile, 'DENZ'+stridx))
        else:
            # If no impurities found reset
            self.zch = None

        # Z data is upside down
        self.zvertp['data'] *=-1.0
        self.zmesh['data'] *=-1.0

        self.NE2Ddata = 0
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i]
            if j != 0: self.NE2Ddata = self.NE2Ddata+1

        self.cells = []
        self.patches = []

        self.row = np.zeros((self.NE2Ddata), dtype=int)
        self.ring = np.zeros((self.NE2Ddata), dtype=int)
        self.rv = np.zeros((self.NE2Ddata, 5))
        self.zv = np.zeros((self.NE2Ddata, 5))
        self.te = np.zeros((self.NE2Ddata))
        self.ti = np.zeros((self.NE2Ddata))
        self.ne = np.zeros((self.NE2Ddata))
        self.ni = np.zeros((self.NE2Ddata))
        self.n0 = np.zeros((self.NE2Ddata))
        self.srec = np.zeros((self.NE2Ddata))
        self.sion = np.zeros((self.NE2Ddata))
        self.imp1_den = np.zeros((len(_imp1_chrg_idx)+1,self.NE2Ddata))
        self.imp2_den = np.zeros((len(_imp2_chrg_idx)+1,self.NE2Ddata))
        ringct = 0        
        k = 0
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i] - 1 # gotcha: convert from fortran indexing to idl/python
            if j >= 0:
                j*=5
                self.rv[k] = [self.rvertp['data'][j],  self.rvertp['data'][j+1], self.rvertp['data'][j+2], self.rvertp['data'][j+3], self.rvertp['data'][j]]
                self.zv[k] = [self.zvertp['data'][j],  self.zvertp['data'][j+1], self.zvertp['data'][j+2], self.zvertp['data'][j+3], self.zvertp['data'][j]]
                self.te[k] = self.teve['data'][i]
                self.ti[k] = self.tev['data'][i]
                self.ni[k] = self.den['data'][i]
                self.ne[k] = self.denel['data'][i]
                self.n0[k] = self.da['data'][i]
                self.srec[k] = self.sirec['data'][i]
                self.sion[k] = self.soun['data'][i]

                # Determine EDGE2D row and ring number (very inefficient but works)
#                self.row[k], self.ring[k] = self.get_eproc_row_ring_from_kval(k)
#                if ringct != self.ring[k]:
#                    print('Reading ring: ', self.ring[k])
#                    ringct = self.ring[k]
                    
                if self.imp1_atom_num:
                    for ichrg in range(len(_imp1_chrg_idx)+1):# +1 to include neutral density
                        self.imp1_den[ichrg,k] = _imp1_denz[ichrg]['data'][i]
                if self.imp2_atom_num:
                    for ichrg in range(len(_imp2_chrg_idx)+1):# +1 to include neutral density
                        self.imp2_den[ichrg,k] = _imp2_denz[ichrg]['data'][i]
                k+=1

        # Read n0 multiplier:
        if True:
            infile = open('/work/bloman/pyproc/n0_multip_files/' + self.tranfile.split('/')[7] + '_n0_multip_inter_ELM.pckl', 'rb')
            n0_multiplier = pickle.load(infile)
            infile.close()
            # Calculate corrected neutral density:
            self.n0 = self.n0*n0_multiplier
            # Zero out neutral density in the main chamber
            zv_cen = self.zv.mean(1)
            mask = zv_cen > -1.3
            self.n0[mask] = 0.

        if False: # This is to leave some small n0 in the main chamber
            infile = open('/work/bloman/pyproc/n0_multip_files/' + self.tranfile.split('/')[7] + '_n0_multip_inter_ELM.pckl', 'rb')
            n0_multiplier = pickle.load(infile)
            infile.close()
            # Calculate corrected neutral density:
            zv_cen = self.zv.mean(1)
            mask = zv_cen > -1.4
            self.n0[np.invert(mask)] = self.n0[np.invert(mask)] * n0_multiplier[np.invert(mask)]
            # Zero out neutral density in the main chamber
            self.n0[mask] = self.n0[mask]

        k = 0
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i] - 1  # gotcha: convert from fortran indexing to idl/python
            if j >= 0:
                j *= 5

                poly = patches.Polygon([(self.rv[k,0], self.zv[k,0]), (self.rv[k,1], self.zv[k,1]), (self.rv[k,2], self.zv[k,2]), (self.rv[k,3], self.zv[k,3]), (self.rv[k,4], self.zv[k,4])], closed=True)
                self.patches.append(poly)
                # create Cell object for each polygon containing the relevant field data
                shply_poly = Polygon(poly.get_xy())

                self.quad_cells.append(process.Cell(self.rmesh['data'][i], self.zmesh['data'][i],
                                       row=self.row[k], ring=self.ring[k],                                       
                                       poly=shply_poly, te=self.te[k], ti=self.ti[k],
                                       ne=self.ne[k], ni=self.ni[k],
                                       n0=self.n0[k], Srec=self.srec[k], Sion=self.sion[k],
                                       imp1_den = self.imp1_den[:,k], imp2_den=self.imp2_den[:,k]))
                k+=1

        ##############################################
        # GET STRIKE POINT COORDS AND SEPARATRIX POLY
        IDL_EprocRing= idl.export_function("EprocRing") # This function is now deprecated!
        rtmp = IDL_EprocRing(self.tranfile,"RMESH","S01",parallel=1)
        ztmp = IDL_EprocRing(self.tranfile,"ZMESH","S01",parallel=1)
        ztmp['ydata'] = -1.0*ztmp['ydata']
        self.osp = [rtmp['ydata'][0], ztmp['ydata'][0]]
        self.isp = [rtmp['ydata'][rtmp['npts']-1], ztmp['ydata'][ztmp['npts']-1]]

        # find start and end index
        istart = -1
        iend = -1
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i] - 1 # gotcha: convert from fortran indexing to idl/python
            j*=5
            if j >= 0:
                if (self.rvertp['data'][j+4] == rtmp['ydata'][0]) and (self.zvertp['data'][j+4] == ztmp['ydata'][0]):
                    istart = i
                if (self.rvertp['data'][j+4] == rtmp['ydata'][rtmp['npts']-1]) and (self.zvertp['data'][j+4] == ztmp['ydata'][ztmp['npts']-1]):
                    iend = i

        sep_points = []
        sep_points_below_xpt = []
        k = 0
        for i in range(istart, iend+1):
            j = self.korpg['data'][i] - 1
            j*=5
            sep_points.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            if self.zvertp['data'][j+0] <= self.geom['zpx']:
                sep_points_below_xpt.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            # sep_points.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            k+=1
        sep_points_below_xpt.append((self.geom['rpx'], self.geom['zpx']))

        self.sep_poly = patches.Polygon(sep_points, closed=False, ec='pink', linestyle='dashed', lw=2.0, fc='None', zorder=10)
        self.shply_sep_poly = Polygon(self.sep_poly.get_xy())
        self.sep_poly_below_xpt = patches.Polygon(sep_points_below_xpt, closed=False, ec='k', fc='None', zorder=10)
        self.shply_sep_poly_below_xpt = Polygon(self.sep_poly_below_xpt.get_xy())

        ##############################################
        # GET WALL POLYGON
        rtmp1 = IDL_EprocDataRead(self.tranfile, 'RVESM1')
        ztmp1 = IDL_EprocDataRead(self.tranfile, 'ZVESM1')
        rtmp2 = IDL_EprocDataRead(self.tranfile, 'RVESM2')
        ztmp2 = IDL_EprocDataRead(self.tranfile, 'ZVESM2')
        # z coord is upside down
        ztmp1['data'] = -1.0*ztmp1['data']
        ztmp2['data'] = -1.0*ztmp2['data']
        # remove non connected segments
        useSegment = np.zeros((rtmp1['npts']))
        nsegs = 0
        for i in range(rtmp1['npts']):
            check = 0
            if i != 0:
                if ((rtmp1['data'][i] == rtmp2['data'][i-1]) and (ztmp1['data'][i] == ztmp2['data'][i-1])) or \
                    ((rtmp2['data'][i] == rtmp1['data'][i-1]) and (ztmp2['data'][i] == ztmp1['data'][i-1])):
                    check = 1
            if i != (rtmp1['npts']):
                if ((rtmp1['data'][i] == rtmp2['data'][i+1]) and (ztmp1['data'][i] == ztmp2['data'][i+1])) or \
                    ((rtmp2['data'][i] == rtmp1['data'][i+1]) and (ztmp2['data'][i] == ztmp1['data'][i+1])):
                    check = 1
            if check:
                useSegment[i] = 1
                nsegs += 1

        wall_poly_pts = []
        for i in range(rtmp1['npts']):
            if useSegment[i]:
                wall_poly_pts.append((rtmp1['data'][i],ztmp1['data'][i]))
                wall_poly_pts.append((rtmp2['data'][i],ztmp2['data'][i]))
        wall_poly_pts.append(wall_poly_pts[0]) # connect last point to first to complete wall polygon

        self.wall_poly = patches.Polygon(wall_poly_pts, closed=False, ec='k', lw=2.0, fc='None', zorder=10)
        self.shply_wall_poly = Polygon(self.wall_poly.get_xy())
        
        

    def get_eproc_param_temp(self, funstr, parstr, par2str=None, args=None):

        idl.execute("""tranfile=' """ + self.tranfile + """ ' """)
        if par2str and args:
            if type(par2str) == int:
                cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ',' + str(par2str) + ',' + args + ')'
            elif type(par2str) == str:
                cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ','  + """'""" + par2str + """'""" + ',' + args + ')'

            idl.execute(cmd)
            idl.execute('xdata=ret.xdata')
            idl.execute('ydata=ret.ydata')
            idl.execute('npts=ret.npts')
            return {'xdata':idl.get('xdata'), 'ydata':idl.get('ydata'), 'npts':idl.get('npts')}
        else:
            cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ')'
            idl.execute(cmd)
            idl.execute('data=ret.data')
            idl.execute('npts=ret.npts')
            return {'data':idl.get('data'), 'npts':idl.get('npts')}



    def get_eproc_row_ring_from_kval(self, kval):

        row_match = -1
        ring_match = -1
            

        idl.execute("""tranfile=' """ + self.tranfile + """/tran""" + """ ' """)

        cmd = 'geom=EprocGeom(tranfile)'
        idl.execute(cmd)
        geom = idl.get('geom')
        nrows = geom['nrows']
        nrings = geom['nrings']
        cmd = 'zmesh=EprocDataRead(tranfile,'+ """'""" + 'ZMESH' + """'""" + ')'
        idl.execute(cmd)

        for irow in range(1,nrows+1):
            cmd = 'kpts_row = EprocRowPts(' + """'""" + str(irow) + """'""" + ',geom, zmesh)'
            idl.execute(cmd)
            kpts_row = idl.get('kpts_row')
            if kval in kpts_row['data']: row_match = irow

        for iring in range(1,nrings+1):
            cmd = 'kpts_ring = EprocRingPts(' + """'""" + str(iring) + """'""" + ',geom)'
            idl.execute(cmd)
            kpts_ring = idl.get('kpts_ring')
            if kval in kpts_ring['data']: ring_match = iring

#        print(row_match, ring_match)
        return row_match, ring_match
    
    # def read_eirene_data(self):

