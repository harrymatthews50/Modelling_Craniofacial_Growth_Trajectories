# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:59:30 2017

@author: harry.matthews
"""
modulepath = 'C:\Users\harry.matthews\Documents\Projects\Modelling_3D_Craniofacial_Growth_Curves_Supp_Material\Modules' #TODO Rememeber to update location on your machine

import sys
sys.path.append(modulepath)
from ShapeStats import  helpers 

import numpy as np
import os
from mayavi import mlab
from scipy.misc import imread
import matplotlib.pyplot as plt
import pandas as pa
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib import rcParams, colorbar

# set default font for matplotlib plotting
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10.
rcParams['legend.fontsize'] = 'small'
        
evaluation_ages = np.array([1.12, 1.5,2., 2.5, 3., 3.5, 4., 4.5, 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14., 14.5, 15., 15.5, 16.])

# render figures offscreen - faster
mlab.options.offscreen = True

 # directoryt to save all figures
gen_datapath = os.path.join(os.path.split(modulepath)[0],'Generated_Data')
part_datapath = os.path.join(os.path.split(modulepath)[0],'Participant_Data')
save_figurepath = os.path.join(gen_datapath,'Figures')           

# create mayavi window
mlabwindow = mlab.figure()
#set lighting and background
helpers.format_mayavi_window(mlabwindow)


# define colour scales for directed distances
dist_vmax= 28.*10**(-5)
dist_vmin = dist_vmax*-1          

#define colour scales for growth vectors
gvec_vmax=7.1*10**(-5)
gvec_vmin = gvec_vmax*-1

#set colormap for heat maps
colormap_name = 'seismic'
#%%

# create separate images of expected faces morphs and growth vectors 
for age in evaluation_ages:
    expected_heads = []
    expected_normals = []
    growth_vectors = []
    for sex in ('Boy','Girl'):
        #load expected 
        expverts, con = helpers.obj2array(os.path.join(gen_datapath,'Expected_Heads_and_morphs','Expected'+sex+'_'+('%.2f'%age).replace('.','_')+'.obj'))
        exaggverts, con = helpers.obj2array(os.path.join(gen_datapath,'Expected_Heads_and_morphs','Exaggerated'+sex+'_'+('%.2f'%age).replace('.','_')+'.obj'))
        expected_heads.append(expverts)
        
        # get normals and alos keep for later
        normals = helpers.get_vertex_normals(expverts.T,con-1,local_flip=True)
        expected_normals.append(normals)
        
        # load growth vectors
        vectors = np.loadtxt(os.path.join(gen_datapath, 'Growth_Vectors','GrowthVectors'+sex+'_'+('%.2f'%age).replace('.','_')+'.txt' ), delimiter=',')
        growth_vectors.append(vectors)
        
        
        
        #plot morphs and exaggerated heads
        for label,verts in zip(('Expected', 'Exaggerated'),(expverts,exaggverts)):
            mesh = mlab.triangular_mesh(verts[:,0],verts[:,1],verts[:,2],con-1,figure=mlabwindow,color = (0.7,0.7,0.7),reset_zoom = False)
            for rotlabel, rotation in zip(('Straight','Three_Quarter', 'Profile'), (helpers.rotate_straight,helpers.rotate_three_quarter,helpers.rotate_profile)):
                rotation(mlabwindow) # rotate image
                #save
                mlab.savefig(os.path.join(save_figurepath,label+sex+rotlabel+('%.2f'%age).replace('.','_')+'.tiff'),size=(1200,1200))
            mesh.remove()
        
        
        
        # Plot growth vector projections in the normal, Lateral, vertical and depth directions
        projections = [np.sum(vectors*normals,axis=0), vectors[0,:],vectors[1,:],vectors[2,:]]
        helpers.rotate_three_quarter(mlabwindow) # rotate to three quarter view
        for direction, values  in zip(('Normal','Lateral','Vertical','Depth'),projections):
            mesh = mlab.triangular_mesh(expverts[:,0],expverts[:,1], expverts[:,2],con-1, scalars = values,vmin=gvec_vmin, vmax = gvec_vmax,colormap = colormap_name,figure = mlabwindow,reset_zoom=False)
            mlab.savefig(os.path.join(save_figurepath,direction+'Growth'+ sex+('%.2f'%age).replace('.','_')+'.tiff'),size=(1200,1200))
            mesh.remove()
        
    # get difference between boy and girl expected heads
    boy, girl = expected_heads
    diff = boy-girl
    
    # get surface normals of girls head
    normals = helpers.get_vertex_normals(girl.T,con-1,local_flip=True)
    
    # get difference in normal, lateral, verrtical and depth directions
    directed_diffs = [np.sum(diff*normals.T,axis=1),diff[:,0],diff[:,1],diff[:,2]]
    
    # plot differences in normal lateral and vertical
    helpers.rotate_three_quarter(mlabwindow) # rotate to three quarter view
    for direction, values in zip(('Normal','Lateral','Vertical','Depth'),directed_diffs):
        mesh = mlab.triangular_mesh(girl[:,0],girl[:,1],girl[:,2],con-1,scalars = values, vmin=dist_vmin,vmax=dist_vmax,colormap = colormap_name,figure=mlabwindow,reset_zoom=False)
        mlab.savefig(os.path.join(save_figurepath,direction+'Difference'+('%.2f'%age).replace('.','_')+'.tiff'),size=(1200,1200))
        mesh.remove()
        
    
    # plot difference in growth rate between groups
    boyvecs, girlvecs = growth_vectors
    gr_diff = np.linalg.norm(boyvecs,axis=0)-np.linalg.norm(girlvecs,axis=0)
    mesh = mlab.triangular_mesh(girl[:,0],girl[:,1],girl[:,2],con-1,scalars = gr_diff, vmin=gvec_vmin,vmax=gvec_vmax,colormap = colormap_name,figure=mlabwindow,reset_zoom=False)
    mlab.savefig(os.path.join(save_figurepath,'GrowthRateDiff'+('%.2f'%age).replace('.','_')+'.tiff'),size=(1200,1200))
    mesh.remove()
    
    #plot angle between growth vectors
    
    #normalise vectors to length 1
    norm_boyvecs = boyvecs/np.linalg.norm(boyvecs,axis=0) 
    norm_girlvecs = girlvecs/np.linalg.norm(girlvecs,axis=0)
    
    cosine = np.sum(norm_boyvecs*norm_girlvecs,axis=0)
    mesh = mlab.triangular_mesh(girl[:,0],girl[:,1],girl[:,2],con-1,scalars = cosine, vmin=-1,vmax=1.,colormap = colormap_name,figure=mlabwindow,reset_zoom=False)
    mlab.savefig(os.path.join(save_figurepath,'GrowthDirectionDiff'+('%.2f'%age).replace('.','_')+'.tiff'),size=(1200,1200))
    mesh.remove()
    
    
#%% Create overall size and shape figure

fig = plt.figure(figsize=(8,4))
gs1=GridSpec(2,4)
gs1.update(wspace=1.,hspace=0.5)


colors = [(0.,0.,0.), (0.7,0.7,0.7)]
linestyles = ['solid','solid']
labels=['Boys','Girls']
handles = []


# plot size and confidence intervals for the regressions of both groups
ax1 = plt.subplot(gs1[:,:2])

# load participant info
metadata = pa.read_excel(os.path.join(part_datapath, 'Participant_Metadata.xlsx'))
size = pa.read_excel(os.path.join(part_datapath, 'Mean_Distance_To_Centroid.xlsx'))

#plot scatter of size vs age for each sex
age = metadata.loc[:,'Age']
sex = metadata.loc[:,'Sex']

boys = metadata.index[sex=='M']
girls = metadata.index[sex=='F']

# plot scatter

boysscat = ax1.scatter(age.loc[boys],size.loc[boys],color=colors[0],s=1.,facecolors='none',label=labels[0])
girlsscat = ax1.scatter(age.loc[girls],size.loc[girls],color=colors[1],s=1.,facecolors='none',label=labels[1])



# plot expectation and confidence intervals for each sex

for i, sex in enumerate(('Boys','Girls')):
    # load info calculated previously
    
    plotinfo = pa.read_excel(os.path.join(gen_datapath,'Overall_size_and_shape_differences_results',sex+'Expected_sizeCI.xlsx'))

    ax1.fill_between(plotinfo.columns,plotinfo.loc['Lower',:], y2=plotinfo.loc['Upper',:], color = colors[i],alpha=0.5, edgecolor='none')
    trend = ax1.plot(plotinfo.columns,plotinfo.loc['Centre',:],linestyle=linestyles[i], color = colors[i],label=labels[i]+'\nTrend')
    

ax1.set_ylabel('Size (mm)')
ax1.set_title('A',weight='bold',size=15.)
legend=ax1.legend(frameon=False, loc='lower right') 


labels=['Boys','Girls']
handles = []

ax2 = plt.subplot(gs1[0,2:])
for i, sex in enumerate(('Boys','Girls')):
    plotinfo = pa.read_excel(os.path.join(gen_datapath,'Overall_size_and_shape_differences_results',sex+'Growth_rateCI.xlsx'))
    ax2.fill_between(plotinfo.columns,plotinfo.loc['Lower',:], y2=plotinfo.loc['Upper',:], color = colors[i],alpha=0.5, edgecolor='none')
    line = ax2.plot(plotinfo.columns,plotinfo.loc['Centre',:],linestyle=linestyles[i], color = colors[i],label=labels[i])[0]
    handles.append(line)
legend = ax2.legend(handles, labels,loc='best',frameon=False)
ax2.set_ylabel('Growth Rate')
ax2.set_title('B',weight='bold',size=15.)


ax3 = plt.subplot(gs1[1,2:])
plotinfo = pa.read_excel(os.path.join(gen_datapath,'Overall_size_and_shape_differences_results','Procrustes_distanceCI.xlsx'))
ax3.fill_between(plotinfo.columns,plotinfo.loc['Lower',:], y2=plotinfo.loc['Upper',:], color = 'k',alpha=0.5, edgecolor='none')
ax3.plot(plotinfo.columns,plotinfo.loc['Centre',:], color = 'k')
ax3.set_ylabel('Procrustes Distance')
ax3.set_title('C',weight='bold',size=15.)


# extra formatting
for ax in [ax2,ax3]:
    ax.set_xlim([1.,16.5])
for ax in [ax1, ax2,ax3]:    
    # allow scientific notation
    ax.ticklabel_format(style='sci',useMathText=True,scilimits=(0.5,1000))
    
    #set visible axes and their positions
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

for ax in [ax2,ax3]:              
    plt.draw()
    ylim = ax.get_ylim()
    ax.set_ylim([0.,ylim[1]])
    
ax1.set_xlabel('Age')
ax3.set_xlabel('Age')
plt.setp(ax2.get_xticklabels(),visible=False)

plt.savefig(os.path.join(save_figurepath,'Overall_size_and_shape_results_figure.jpg'),bbox_inches = 'tight',dpi = 800)
plt.savefig(os.path.join(save_figurepath,'Overall_size_and_shape_results_figure.tiff'),bbox_inches = 'tight',dpi = 800)
plt.savefig(os.path.join(save_figurepath,'Overall_size_and_shape_results_figure.eps'),bbox_inches = 'tight')



#%% Compile shape differences and growth patterns into figures for each direction

fig = plt.figure(figsize=(5,9))
gs1 = GridSpec(9,6)
gs1.update(wspace=0,hspace=0)
plot_ages = [1.12, 2., 4., 6., 8., 10., 12., 14., 16.]

# labels for colorbars
directions = ['Normal','Lateral','Vertical','Depth']
direction_descriptions = [('Outward','Inward'),('Right','Left'), ('Superior','Inferior'),('Anterior','Posterior')]

#some formatting for colorbar
cbar_height = 0.015
horizontal_pad = 0.01
vertical_pad = 0.01


column_labels = ['Boys','Girls','Difference','Boys\nPattern','Girls\nPattern']
colnums = [0,1,2,4,5] # which columns to plot them in

#set column labels
for label, col in zip(column_labels,colnums):
    plt.subplot(gs1[0,col]).set_title(label, fontsize=8.)          
                   
#set row labels
for row, age in enumerate(plot_ages):
    plt.subplot(gs1[row,0]).set_ylabel('%.2f'%age,fontsize=8.)
for direction_num, direction in enumerate(directions):
    for row, age in enumerate(plot_ages):
        #list of paths to the images to be plotted in this row
        imagepaths = [os.path.join(save_figurepath,'ExpectedBoyThree_Quarter'+('%.2f'%age).replace('.','_')+'.tiff'),
                      os.path.join(save_figurepath,'ExpectedGirlThree_Quarter'+('%.2f'%age).replace('.','_')+'.tiff'),
                      os.path.join(save_figurepath,direction+'Difference'+('%.2f'%age).replace('.','_')+'.tiff'), 
                      os.path.join(save_figurepath,direction+'GrowthBoy'+('%.2f'%age).replace('.','_')+'.tiff'),
                    os.path.join(save_figurepath,direction+'GrowthGirl'+('%.2f'%age).replace('.','_')+'.tiff')]
        
        
        for impath, col in zip(imagepaths,colnums):
            #load and crop image
            im = imread(impath)
            im = helpers.crop_image(im,left=0.2,right=0.2,top=0.15,bottom=0.025)
            ax = plt.subplot(gs1[row,col])
            ax.imshow(im)
            helpers.invisible_axes(ax)
            ax.axis("equal")
        
        
    #add colorbars
    ax = plt.subplot(gs1[8,2])
    axbox = ax.get_position() # position first colorbar under 'Difference map'
    cax1 = fig.add_axes([axbox.xmin+horizontal_pad,axbox.ymin-cbar_height-vertical_pad, (axbox.xmax-axbox.xmin)-2*horizontal_pad,cbar_height])
    cbar1 = colorbar.ColorbarBase(cax1,cmap=plt.get_cmap(colormap_name),norm=plt.Normalize(vmin=dist_vmin,vmax = dist_vmax),orientation = 'horizontal')
    cbar1.outline.set_edgecolor('None')
    cbar1.outline.set_facecolor('None')
    cbar1.set_ticks([dist_vmin,dist_vmax])
    cbar1.set_ticklabels(('More\n'+direction_descriptions[direction_num][1],'More\n'+direction_descriptions[direction_num][0]))
    
    cax1.tick_params(labelsize=5.)
   
    cax1.tick_params(length=0.)
    
    # position second colorbar so that it spans both pattern maps
    pos1 = plt.subplot(gs1[8,4]).get_position()
    pos2 = plt.subplot(gs1[8,5]).get_position()
    xmin = pos1.xmin
    xmax = pos2.xmax
    
    cax2 = fig.add_axes([xmin+horizontal_pad,pos1.ymin-cbar_height-vertical_pad, xmax-xmin-2*horizontal_pad,cbar_height])
    
    cbar2 = colorbar.ColorbarBase(cax2,cmap=plt.get_cmap(colormap_name),norm=plt.Normalize(vmin = gvec_vmin,vmax = gvec_vmax),orientation = 'horizontal')
    cbar2.outline.set_edgecolor('None')

    cbar2.outline.set_facecolor('None')

    cbar2.set_ticks([gvec_vmin,gvec_vmax])
    cbar2.set_ticklabels(('Moving\n'+direction_descriptions[direction_num][1],'Moving\n'+direction_descriptions[direction_num][0]))
    cax2.tick_params(labelsize=5.)
    cax2.tick_params(length=0.)
    plt.savefig(os.path.join(save_figurepath, direction+'Shape_and_Growth_Differences.tiff'),dpi=800, bbox_inches = 'tight')

#%% Plot figure comparing growth rate and growth direction                

plot_ages = [1.12, 2., 4., 6., 8., 10., 12., 14., 16.]

fig = plt.figure(figsize=(7,2))
gs1 = GridSpec(2,len(plot_ages))
gs1.update(wspace=0,hspace=0)


# parameters for colorbar
cbar_width = 0.015
horizontal_pad = 0.01
vertical_pad = 0.02

# set column labels
for col, age in enumerate(plot_ages):
    plt.subplot(gs1[0,col]).set_title('%.2f'%age,fontsize=8.)

rowlabels = ['Rate\nDifference', 'Direction\nDifference']
for row, label in enumerate(rowlabels):
    plt.subplot(gs1[row,0]).set_ylabel(label, fontsize=8.)
    
for col, age in enumerate(plot_ages):
    rate_diff_im = helpers.crop_image(imread(os.path.join(save_figurepath,'GrowthRateDiff'+('%.2f'%age).replace('.','_')+'.tiff')), left=0.2,right=0.2,top=0.15,bottom=0.025)
    direction_diff_im = helpers.crop_image(imread(os.path.join(save_figurepath,'GrowthDirectionDiff'+('%.2f'%age).replace('.','_')+'.tiff')), left=0.2,right=0.2,top=0.15,bottom=0.025)
    
    rate_diff_ax = plt.subplot(gs1[0,col])
    dir_diff_ax = plt.subplot(gs1[1,col])
    
    rate_diff_ax.imshow(rate_diff_im)
    dir_diff_ax.imshow(direction_diff_im)
    
    helpers.invisible_axes(rate_diff_ax)
    helpers.invisible_axes(dir_diff_ax)
    
# add colorbars
ax = plt.subplot(gs1[0,8])
axbox = ax.get_position()
cax1 = fig.add_axes([axbox.xmax+horizontal_pad,axbox.ymin+vertical_pad,cbar_width,(axbox.ymax-axbox.ymin)-2*vertical_pad] )
cax1.tick_params(labelsize=5.)  
cax1.tick_params(length=0.)

cbar1 = colorbar.ColorbarBase(cax1,cmap=plt.get_cmap(colormap_name),norm=plt.Normalize(vmin = gvec_vmin,vmax = gvec_vmax),orientation = 'vertical')
cbar1.outline.set_edgecolor('None')
cbar1.outline.set_facecolor('None')
cbar1.set_ticks([gvec_vmin*0.8,gvec_vmax*0.8]) # position labels slightly in from extremes of colorbar, otherwise looks ugly
cbar1.set_ticklabels(('Girls\nFaster', 'Boys\nFaster'))


ax = plt.subplot(gs1[1,8])
axbox = ax.get_position()
cax2 = fig.add_axes([axbox.xmax+horizontal_pad,axbox.ymin+vertical_pad,cbar_width,(axbox.ymax-axbox.ymin)-2*vertical_pad] )
cax2.tick_params(labelsize=5.)  
cax2.tick_params(length=0.)

cbar2 = colorbar.ColorbarBase(cax2,cmap=plt.get_cmap(colormap_name),norm=plt.Normalize(vmin=-1., vmax = 1.),orientation = 'vertical')
cbar2.outline.set_edgecolor('None')
cbar2.outline.set_facecolor('None')
cbar2.set_ticks([-0.8,0.8])
cbar2.set_ticklabels(('Opposite\nDirection', 'Same\nDirection'))

plt.savefig(os.path.join(save_figurepath,'Direction_and_rate_differences.tiff'),dpi=800,bbox_inches = 'tight')

    
#%% Create plot of morphs

column_labels = ['Exaggerated\nBoys','Expected\nBoys','Expected\nGirls','Exaggerated\nGirls','Exaggerated\nBoys','Expected\nBoys','Expected\nGirls','Exaggerated\nGirls']
colnums = [0,1,2,3,5,6,7,8]
plot_ages = [1.12, 2., 4., 6., 8., 10., 12., 14., 16.]

fig = plt.figure(figsize=(8,10))
gs1 = GridSpec(9,9)
gs1.update(wspace=0,hspace=0)

#set column labels
for label, col in zip(column_labels,colnums):
    plt.subplot(gs1[0,col]).set_title(label, fontsize=8.)          
                   
#set row labels
for row, age in enumerate(plot_ages):
    plt.subplot(gs1[row,0]).set_ylabel('%.2f'%age,fontsize=8.)




for row, age in enumerate(plot_ages):
    
    
    imagepaths = [os.path.join(save_figurepath,'ExaggeratedBoyStraight'+('%.2f'%age).replace('.','_')+'.tiff'),
              os.path.join(save_figurepath,'ExpectedBoyStraight'+('%.2f'%age).replace('.','_')+'.tiff'),
              os.path.join(save_figurepath,'ExpectedGirlStraight'+('%.2f'%age).replace('.','_')+'.tiff'),
              os.path.join(save_figurepath,'ExaggeratedGirlStraight'+('%.2f'%age).replace('.','_')+'.tiff'),
              
              os.path.join(save_figurepath,'ExaggeratedBoyProfile'+('%.2f'%age).replace('.','_')+'.tiff'),
              os.path.join(save_figurepath,'ExpectedBoyProfile'+('%.2f'%age).replace('.','_')+'.tiff'),
              os.path.join(save_figurepath,'ExpectedGirlProfile'+('%.2f'%age).replace('.','_')+'.tiff'),
              os.path.join(save_figurepath,'ExaggeratedGirlProfile'+('%.2f'%age).replace('.','_')+'.tiff'),
                        ]
    for impath, col in zip(imagepaths,colnums):
            #load and crop image
            im = imread(impath)
            im = helpers.crop_image(im,left=0.2,right=0.05,top=0.15,bottom=0.025)
            ax = plt.subplot(gs1[row,col])
            ax.imshow(im)
            helpers.invisible_axes(ax)
            ax.axis("equal")
plt.savefig(os.path.join(save_figurepath,'Morph_plot.tiff'),dpi = 800, bbox_inches = 'tight')

     
#%% Make animation of expected faces and difference


fig = plt.figure(figsize=(11,4.))
gs1 = GridSpec(1,3)

axes = [plt.subplot(gs1[0,i]) for i in range(3)]
for ax in axes:    
    helpers.invisible_axes(ax)
axlabels = ['Boys','Girls','Shape\nDifference']

fig_adjust = 0.
cbar_width = 0.015
horizontal_pad = 0.01
vertical_pad = 0.02
diff_description = [('Outward','Inward'),('Right','Left'), ('Superior','Inferior'),('Anterior','Posterior')]


#add colorbar
ax = axes[2]
axbox = ax.get_position()
cax1 = fig.add_axes([axbox.xmax+horizontal_pad,axbox.ymin+vertical_pad,cbar_width,(axbox.ymax-axbox.ymin)-2*vertical_pad] )
cax1.tick_params(labelsize=12.) 
cax1.tick_params(length=0.)

cbar = colorbar.ColorbarBase(cax1,cmap=plt.get_cmap('seismic'),norm=plt.Normalize(vmin=dist_vmin,vmax=dist_vmax),orientation = 'vertical')
cbar.outline.set_edgecolor('None')
cbar.outline.set_facecolor('None')
cbar.set_ticks([dist_vmin*0.8,dist_vmax*0.8])




for diffnum,direction in enumerate(['Normal']):#,'Lateral','Vertical','Depth']):
            
            animation_writer = animation.writers['ffmpeg'](fps=1)

            with animation_writer.saving(fig, os.path.join(save_figurepath,'Expectation_and_'+direction+'Difference.mp4'), 100):
                for age in evaluation_ages:
                    
                                fnames = [os.path.join(save_figurepath,'ExpectedBoyThree_Quarter'+('%.2f'%age).replace('.','_')+'.tiff'),
                                          os.path.join(save_figurepath,'ExpectedGirlThree_Quarter'+('%.2f'%age).replace('.','_')+'.tiff'),
                                          os.path.join(save_figurepath,direction+'Difference'+('%.2f'%age).replace('.','_')+'.tiff')       
                                         ]
                                cbar.set_ticklabels(('More\n'+diff_description[diffnum][1],'More\n'+diff_description[diffnum][0]))
                                for i, fn in enumerate(fnames):
                                    im = helpers.crop_image(imread(fn),left=0.2,right=0.2,top=0.15,bottom=0.025)
                                    axes[i].imshow(im)
                                    axes[i].axis("equal")
                                    axes[i].set_xlabel(axlabels[i], size=16.)
                                
                                fig.suptitle('Age='+'%.2f'%age,size=20.)
                

                                animation_writer.grab_frame()


#%% Plot distributions of male/female scores 
fig = plt.figure(figsize=(9,3))
gs1 = GridSpec(1,4)
age_bin_limits = [0,5.,10.,15.,20.]

titles = ['<5','5-10','10-15','>15']

# load scores
scores = pa.read_excel(os.path.join(gen_datapath,'Classification', 'Scores.xlsx'))

# load participant metadata
metadata = pa.read_excel(os.path.join(part_datapath, 'Participant_Metadata.xlsx'))

age_bin_code = np.digitize(metadata.loc[:,'Age'].as_matrix(),age_bin_limits)-1 # 
males = metadata.loc[:,'Sex_numeric'].as_matrix()==1.

for age_bin in np.unique(age_bin_code):
    binmask = age_bin_code==age_bin
    ax = fig.add_subplot(gs1[0,age_bin])
    
    binscores = scores.as_matrix()[binmask]
    min_score = np.min(binscores)
    max_score = np.max(binscores)
    lim = np.max(np.abs([min_score,max_score]))
    width = lim/30
    bins = np.linspace(-lim,lim,15)
    centres = np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)])

    M_frequ, dump = np.histogram(binscores[males[binmask]].astype(float),bins=bins)
    F_frequ, dump = np.histogram(binscores[males[binmask]==False].astype(float),bins=bins)
    ax.bar(centres-width,M_frequ,width,color = 'k',label='Boys',edgecolor='none')
    ax.bar(centres,F_frequ,width,color = (0.7,0.7,0.7),label='Girls',edgecolor='none')

    ax.set_ylim([0.,40])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom') 
    ax.set_xlabel('Score')
    ax.set_title(titles[age_bin])
    if age_bin == 3:
       legend = ax.legend(frameon=False)
    
    if age_bin==0:
        ax.set_ylabel('Count')
    else:
        plt.setp(ax.get_yticklabels(),visible=False)
       
plt.savefig(os.path.join(save_figurepath,'Score_histograms.jpg'),dpi = 800,bbox_inches = 'tight')
plt.savefig(os.path.join(save_figurepath,'Score_histograms.eps'),bbox_inches = 'tight')

#%% Plot distribution of ages for each sex

#load participant metadata
metadata = pa.read_excel(os.path.join(part_datapath, 'Participant_Metadata.xlsx'))
sex = metadata.loc[:,'Sex_numeric'].as_matrix()
ages = metadata.loc[:,'Age'].as_matrix()

malemask = sex==1.
femalemask = sex==2.
bins=[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.]
centres = np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)])

M_frequ, dump = np.histogram(ages[malemask].astype(float),bins=bins)
F_frequ, dump = np.histogram(ages[femalemask].astype(float),bins=bins)

N = len(M_frequ)

width = 0.35
fig = plt.figure(figsize=(3,3))
ax = plt.subplot()
mbar = ax.bar(centres-width,M_frequ,width,color='k')[0]

fbar = ax.bar(centres,F_frequ,width,color=(0.7,0.7,0.7))[0]

ax.set_xticks([bins[i] for i in range(0,len(bins),2)])
#ax.set_xticklabels([str(item) for item in  centres],rotation='vertical')
ax.set_xlabel('Age')
ax.set_ylabel('N')
ax.set_ylim([0,50])
legend = ax.legend((mbar,fbar),('Males','Females'),loc='best',frameon=False,borderaxespad=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.setp(ax.get_yticklabels())
plt.setp(ax.get_xticklabels())

plt.savefig(os.path.join(save_figurepath,'Age_distribution'+'.png'),dpi=800,bbox_inches='tight',transparent=True)
plt.savefig(os.path.join(save_figurepath,'Age_distribution'+'.jpg'),dpi=800,bbox_inches='tight',transparent=False)
plt.savefig(os.path.join(save_figurepath,'Age_distribution'+'.eps'),dpi=800,bbox_inches='tight')






