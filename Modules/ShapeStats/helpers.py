# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:41:24 2017

@author: harry.matthews
"""

import collections
import os
import numpy as np


import mayavi
import pandas as pa
import csv

        
def writewobj(verts,con,fn):
    """
    Writes vertices and connectivity matrix to .obj file at fn
    """
    if os.path.splitext(fn)[1]!='.obj':
        raise ValueError('Filename does not have \'.obj\' extension')
    
    with open(fn,'wb') as csvfile:
        writerobj = csv.writer(csvfile, delimiter=' ')
        for i in range(verts.shape[1]):
            row = ['v'] + [str(item) for item in verts[:,i]]
            writerobj.writerow(row)

        for i in range(con.shape[0]):
                    row = ['f'] + [str(item) for item in con[i,:]]
                    writerobj.writerow(row)

    
def PC_space_to_landmark_space(PCcoefs,eigenvectors, meanvector=None):
    if meanvector is None:
        meanvector = np.zeros(eigenvectors.shape[1])
    
    #back_project
    landmark_array = np.dot(PCcoefs,eigenvectors) + meanvector
    
    #reshape to 3 x n(vertices) x k(observations)
    landmark_array = np.reshape(landmark_array,[landmark_array.shape[0],3,landmark_array.shape[1]/3],order='F').transpose(1,2,0)
    return landmark_array





def mynormpdf(x, sigma = 1, mu = 0):

    z = (x-mu)/float(sigma)
    y = np.exp((-z**2)*0.5)/np.sqrt(2*np.pi)
    return y




def configure_input_matrices(X, Y,indices):
        
        if isinstance(X, pa.Series):
            X = pa.DataFrame(X)
        if isinstance(Y, pa.Series):
            Y = pa.DataFrame(Y)
    
        if isinstance(X, pa.DataFrame)==False:
            raise TypeError('X-block should be a pandas DataFrame')
        if isinstance(Y, pa.DataFrame)==False:
            raise TypeError('Y-block should be a pandas DataFrame')    
        


        
        #Check for duplicates
        y_duplicates = [item for item, count in collections.Counter(Y.index).items() if count > 1]
        x_duplicates = [item for item, count in collections.Counter(X.index).items() if count > 1]
        
        if any([len(x_duplicates)>0,len(y_duplicates)>0]):
            raise ValueError('Duplicate values of '+str(y_duplicates)+ 'in Y-block Index \n Duplicate values of '+str(x_duplicates)+ 'in X-block Index')
        
        
        # check for lack of correspondence between X and Y blocl
        if isinstance(indices,str):
            if indices=='All':    
                missingy = [item for item in X.index if item not in Y.index]
                missingx = [item for item in Y.index if item not in X.index]
        else:
            # check for any indices that do not have a corresponding value and are to be used (i.e. are listed in indices)
            missingy = [item for item in X.index if all([item not in Y.index, item in indices])]
            missingx = [item for item in Y.index if all([item not in X.index, item in indices])]
        
        if any([len(missingx)>0,len(missingy)>0]):
                raise ValueError('Unable to find observations '+str(missingy)+ 'in Y-block Index \n Unable to find observations '+str(missingx)+ 'in X-block Index')
        
        
        
        
        # trim to included indices only, if necessary
        if isinstance(indices,str)==False: 
                X = X.loc[indices,:]
                Y = Y.loc[indices,:] # do not ever cast this into a list, because it re-orders the elements and this causes problems with corresponding to weights in regression functions that use this function
        
        
        
        
        
        
        
        
        # check for non numeric values in datadrame
        try:
            Xmat = np.atleast_2d(X.as_matrix().astype(float))
        except:
            raise ValueError('unable to convert X-block into an array of floats, Check for cells containg text')
            
        try:
            Ymat = np.atleast_2d(Y.as_matrix().astype(float))
        except:
            raise ValueError('unable to convert Y-block into an array of floats, Check for cells containg text')
       
        
        if np.sum(np.isnan(Xmat))>0:
            raise TypeError('has encountered an NaN value in the X-block. This is most likely reflects missing data')
        if np.sum(np.isnan(Ymat))>0:
            raise TypeError('has encountered an NaN value in the Y-block. This is most likely reflects missing data')
        
        # if column vector, make row
            if Xmat.shape[0]==1:
                Xmat = Xmat.T
        # if column vector, make row
            if Ymat.shape[0]==1:
                Ymat = Ymat.T
        
        return X, Y, Xmat, Ymat



            
def format_mayavi_scalarbar(scene):
        module_manager = scene.children[0].children[0].children[0]
        if isinstance(module_manager,mayavi.core.module_manager.ModuleManager)==False:
            module_manager = scene.children[0].children[0]
        
        module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([ 0.78866841,  0.15208334])
        module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = np.array([100000, 100000])


        module_manager.scalar_lut_manager.title_text_property.shadow_offset = np.array([ 1, -1])
        module_manager.scalar_lut_manager.title_text_property.italic = False

        module_manager.scalar_lut_manager.label_text_property.shadow_offset = np.array([ 1, -1])
        module_manager.scalar_lut_manager.label_text_property.font_family = 'arial'
        module_manager.scalar_lut_manager.label_text_property.shadow_offset = np.array([ 1, -1])
        module_manager.scalar_lut_manager.label_text_property.color = (0.0, 0.0, 0.0)
        module_manager.scalar_lut_manager.label_text_property.shadow_offset = np.array([ 1, -1])
        module_manager.scalar_lut_manager.label_text_property.italic = False
        module_manager.scalar_lut_manager.label_text_property.font_size = 5
        module_manager.scalar_lut_manager.title_text_property.font_size = 5



        #Show Colourbar
        module_manager.scalar_lut_manager.show_scalar_bar = True
        module_manager.scalar_lut_manager.show_legend = True


def format_mayavi_window(scene):
    scene.scene.background = (1.0, 1.0, 1.0)
    scene.scene.z_plus_view()
    # ambientish lighting
    from tvtk.pyface.light_manager import CameraLight
    camera_light4 = CameraLight(scene.scene)
    scene.scene.light_manager.light_mode = 'vtk'
    #camera_light4 = CameraLight(scene)
    scene.scene.light_manager.lights[4:4] = [camera_light4]
    scene.scene.light_manager.number_of_lights = 5

    azims = [0,-45,45,-90,90]
    for l in range(5):
        light = scene.scene.light_manager.lights[l]
        light.activate=True
        light.intensity=0.4
        light.azimuth = azims[l]
        light.elevation = 0

    


    
    scene.scene.render()


def rotate_three_quarter(scene):
    scene.scene.camera.position = [-0.019731792822820279, -0.0026287531664291423, 0.035864322463504274]
    scene.scene.camera.focal_point = [-4.7538551304999663e-05, 0.00043069758460500011, -0.00051276164985000023]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.0029590594905338358, 0.99634257677142046, 0.085397386841855286]
    scene.scene.camera.clipping_range = [0.024178586085343901, 0.063346209557328698]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

def rotate_profile(scene):
    scene.scene.camera.position = [-0.039985833160833713, -0.00035065706455172138, 0.010642706171288249]
    scene.scene.camera.focal_point = [-4.7538551304999663e-05, 0.00043069758460500011, -0.00051276164985000023]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.035003636954052675, 0.98037951863445871, 0.19398645530566372]
    scene.scene.camera.clipping_range = [0.026868000044897811, 0.059960791482613775]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

def rotate_straight(scene):
    scene.scene.z_plus_view()

        
def get_centroid_size(vertices):
    #assuming a 3D mesh
    if vertices.shape[0]!=3: 
        raise ValueError("centroid size takes a 3 x n_vertices array")
    
    v0 = vertices-np.atleast_2d(np.mean(vertices,axis=1)).T
    dists = np.linalg.norm(v0,axis=0)
    return np.sqrt(np.sum(dists**2))

def get_mesh_radius(vertices):
    if vertices.shape[0]!=3: 
        raise ValueError("mesh_radius takes a 3 x n_vertices array")
    v0 = vertices-np.atleast_2d(np.mean(vertices,axis=1)).T
    dists = np.linalg.norm(v0,axis=0)
    return np.mean(dists)

def scale_to_centroid_size(vertices, centroid_size=1.):
    if vertices.shape[0]!=3: 
        raise ValueError("scale to centroid size takes a 3 x n_vertices array")
    centroid = np.atleast_2d(np.mean(vertices,axis=1)).T
    v0 = vertices-centroid
    scalefactor = centroid_size/get_centroid_size(vertices)
    scaled_v0 = v0*scalefactor
    out = scaled_v0+centroid
    assert (get_centroid_size(out)-centroid_size<.0000001)
    return out

def invisible_axes(ax):
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            return ax

def crop_image(im, left=0.2,right=0.23,top=0.15,bottom=0.15):
    Ypix = im.shape[0]
    Xpix = im.shape[1]

    startX = int(left*Xpix)
    stopX =  int(1-right*Xpix)

    startY  =   int(top*Ypix)
    stopY= int(1-bottom*Ypix)

    return im[startY:stopY,startX:stopX]


def  obj2array(fn):

        if type(fn)==str:
                if os.path.isfile(fn)==False:
                        raise ValueError('obj2array: unable to locate file ' + str(fn))
                fobj =open(fn)

        vertices = list()
        connectivity = list()
        uv = list()
        
        fcount = 0

        for l in fobj:
                    line = l.rstrip('\n')
                    data = line.split()
                    if len(data)==0:
                            pass
                    else:
                            
                            if data[0] == 'v':
                                vertices.append(np.atleast_2d(np.array([float(item) for item in data[1:4]])))
                            
                            elif data[0]=='vt':
                                uv.append(np.atleast_2d(np.array([float(item) for item in data[1:3]])))


                            elif data[0]=='f':
                                    

                                if fcount == 0:    #on first face establish face format

                                    fcount = fcount + 1
                                    if data[1].find('/')==-1:  #Case 1
                                        case = 1
                            #            v = zeros([1,len(data)-1])
                                    elif data[1].find('//')==True:
                                        case = 4
                             #           v = zeros([1,len(data)-1])
                              #          vn = zeros([1,len(data)-1])
                                    elif len(data[1].split('/'))==2:
                                        case = 2
                            #            v = zeros([1,len(data)-1])
                            #           vt = zeros([1,len(data)-1])
                                    elif len(data[1].split('/'))==3:
                                        case = 3
                             #           v = zeros([1,len(data)-1])
                            #          vt = zeros([1,len(data)-1])
                             #           vn = zeros([1,len(data)-1])
                                    
                             

                                if case == 1:
                                    f = np.atleast_2d([int(item) for item in data[1:len(data)]])
                                    connectivity.append(f)

                                if case == 2:
                                    splitdata = [item.split('/') for item in data[1:len(data)]]
                                    f = np.atleast_2d([int(item[0]) for item in splitdata])
                                 #   vt.append(nm.atleast_2d([int(item[1]) for item in splitdata]))

                                #         textindex = vt

                                    connectivity.append(f)
                                  #       textindex = append(textindex,vt,axis=0)

                                if case == 3:
                                    splitdata = [item.split('/') for item in data[1:len(data)]]
                                    f = np.atleast_2d([int(item[0]) for item in splitdata])
                                    #vt.append(nm.atleast_2d([int(item[1]) for item in splitdata]))
                                   # vn = [int(item[2]) for item in splitdata]

                                    connectivity.append(f)
                                     #    textindex = append(textindex,vt,axis=0)
                                      #   normalindex = append(normalindex,n,axis=0)
                                         #print(len(normalindex))

                                if case == 4:
                                    splitdata = [item.split('//') for item in data[1:len(data)]]
                                    f = np.atleast_2d([int(item[0]) for item in splitdata])
                                   # vn = [int(item[1]) for item in splitdata]
                   
                                    connectivity.append(f)



        vertices = np.concatenate(vertices, axis = 0)
        if len(uv)==0:
            uv=None
        else:
            uv = np.concatenate(uv, axis = 0)
        
        


            


        conarray = np.concatenate(connectivity, axis=0)
                        

        return vertices, conarray      
        
            
def get_facet_normals(verts,con):
    p0 = verts[:,con[:,0]]
    p1 = verts[:,con[:,1]]
    p2 = verts[:,con[:,2]]
    # for each face get two edges
    e0 = p2-p1
    e1 = p0-p2

    Fnormals = np.cross(e0.T,e1.T).T

    # make unit length
    Fnormals/=np.atleast_2d(np.linalg.norm(Fnormals,axis=0))
    
    
    return Fnormals

def local_flip_normals(normals,locs):
    """Flips normals so they point away from the centroid of the configuration defined by locs """
    # get projectiuon of vector between centroid of points and normals to determine if pointing inward or outward
    # Will not work proiperly on mandibles
    #TODO this should probably be based on local neighbourhood of vertices rather  than all points
    
    vecs = locs-np.atleast_2d(np.mean(locs, axis=1)).T
    projs = np.sum(vecs*normals,axis=0)
    scalar = np.atleast_2d(np.sign(projs))# either 1 or -1 to flip required normals
    flipped_normals=normals*scalar 
    return flipped_normals

def global_flip_normals(normals, locs):
    """Determines if more than half of the normals point towards the centroid of the configuration defined by locs
    if so, it will flip the direction of all the normals"""
    vecs = locs-np.atleast_2d(np.mean(locs, axis=1)).T
    projs = np.sum(vecs*normals,axis=0)
    if np.sum(projs<0.)>projs.size/2:
        return normals*-1
    else:
        return normals

def get_vertex_normals(verts_array,con, global_flip=False,local_flip=False):
    #TODO It should be possible to do this without the loop
    """Compute the normals at each vertex
    INPUTS:
        verts_array - a 3 x n(vertices) array of vertex co-oridinates or a 3 x n(vertices) x p(observations) array of corresponding vertex co-ordinates
        con - a n(facets) x 3 connectivity matrix defining the connectivity of the mesh
        
    OUTPUTS:
        Vnormals - a 3 x n(vertices) x n_observations array of vertex normals if if verts_array was three-dimensional, otheriwise it will be a 3 x n(vertices) array of vertex normals
        """
    if verts_array.shape[0]!=3:
        raise ValueError('First dimension of vertices array should have length 3 corresponding to X, Y and Z axes but first dimension has length ' + str(verts_array.shape[0]))
    if con.shape[1]!=3:
        raise ValueError('Second dimension of connectivity array should have length 3 corresponding the three points on each facet but second dimension has length ' + str(con.shape[1]))
    
    if np.min(con)!=0:
        raise ValueError('The mininum value in the connectivity matrix should be 0 but is '+ str(np.min(con)))
    
    verts_array = np.atleast_3d(verts_array)
    normals_array = np.zeros_like(verts_array)
    for mesh_num in xrange(verts_array.shape[2]):
        verts=verts_array[:,:,mesh_num]
        N = get_facet_normals(verts,con)
        verts = verts.T #ToDO this transposition is unnecessary if i fix the code below
        N = N.T
        # get edges vectors
        e0 = verts[con[:,2]]-verts[con[:,1]]
        e1 = verts[con[:,0]]-verts[con[:,2]]
        e2 = verts[con[:,1]]-verts[con[:,0]]
    
    
    
        # edge length
        de0 = np.linalg.norm(e0,axis=1)
        de1 = np.linalg.norm(e1,axis=1)
        de2 = np.linalg.norm(e2,axis=1)
    
    
    
        #face area
        Af = 0.5*np.apply_along_axis(np.linalg.norm,1,np.cross(e0,e1))
    
    
    
    
        Vnormals = np.zeros(verts.shape)
    
        for i in range(con.shape[0]):
            #weight according to area and edge length
            wfv0 = Af[i]/(np.power(de1[i],2)*np.power(de2[i],2))
            wfv1 = Af[i]/(np.power(de0[i],2)*np.power(de2[i],2))
            wfv2 = Af[i]/(np.power(de1[i],2)*np.power(de0[i],2))
    
    
            Vnormals[con[i,0],:] = Vnormals[con[i,0],:] + wfv0*N[i,:]
            Vnormals[con[i,1],:] = Vnormals[con[i,1],:] + wfv1*N[i,:]
            Vnormals[con[i,2],:] = Vnormals[con[i,2],:] + wfv2*N[i,:]
    
        Vnormals = Vnormals/np.atleast_2d(np.linalg.norm(Vnormals,axis=1)).T
        if local_flip==True:
            Vnormals = local_flip_normals(Vnormals,verts)
        if global_flip==True:
            Vnormals = global_flip_normals(Vnormals,verts)
        normals_array[:,:,mesh_num] = Vnormals.T
    return np.squeeze(normals_array)            


