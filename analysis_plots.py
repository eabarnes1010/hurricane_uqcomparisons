import matplotlib.pyplot as plt
import numpy as np
import shash
import model_diagnostics

# import imp
# imp.reload(model_diagnostics)

colors = ('#284E60','#E1A730','#D95980','#C3B1E1','#351F27','#A9C961')
clr_shash = colors[0]
clr_bnn   = colors[3]
clr_truth = 'dimgray'



FS = 16
### for white background...
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='white')
plt.rc('axes',facecolor='white')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('xtick',color='dimgrey')
plt.rc('ytick',color='dimgrey')

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([])  

def plot_sample(ax, onehot_val, shash_incs, shash_cpd, bnn_cpd, sample=130):
    plt.sca(ax)  

    if(shash_cpd.shape[0]<sample):
        sample = shash_cpd.shape[0]-1
    
    bins = np.arange(np.min(shash_incs),np.max(shash_incs)+2,2)

    # results for SHASH
    plt.plot(shash_incs,
             shash_cpd[sample,:],
             color=clr_shash,
             linewidth=4,
             label='SHASH',
            )

    # results for BNN
    if bnn_cpd is not None:
        plt.hist(bnn_cpd[sample,:],
                 bins=bins,
                 histtype=u'step',
                 density=True, 
                 color=clr_bnn,
                 linewidth=2,
                 label='BNN',
                )

    # truth
    plt.axvline(x=onehot_val[sample,0],color=clr_truth,linestyle='--')
    plt.text(x=onehot_val[sample,0]+1, 
             y = .09, 
             s = 'truth', 
             color=clr_truth,
             fontsize=12,
             horizontalalignment='left',
            )
    plt.legend()
    
    plt.ylim(0,0.1)
    plt.xlim(-45,45)

    ax = plt.gca()
    xticks = ax.get_xticks()
    yticks = np.around(ax.get_yticks(),2)
    ax.set_xticks(xticks.astype(int),xticks.astype(int))
    ax.set_yticks(yticks,yticks)

    plt.title('validation sample ' + str(sample))
    plt.xlabel('predicted deviation from consensus (knots)')
    plt.ylabel('probability density function')    

    
    
    
def plot_pits(ax, x_val, onehot_val, model_shash, shash_cpd, bnn_cpd):
    plt.sca(ax)      
    
    # shash pit
    bins, hist_shash, D_shash, EDp_shash = model_diagnostics.compute_pit('shash',onehot_val, x_data=x_val,model_shash=model_shash)
    bins_inc = bins[1]-bins[0]

    if bnn_cpd is not None:
        bin_add = bins_inc/6+bins_inc/6 
        bin_width = bins_inc/3
    else:
        bin_add = bins_inc/2
        bin_width = bins_inc*.75
    plt.bar(hist_shash[1][:-1] + bin_add,
             hist_shash[0],
             width=bin_width,
             color=clr_shash,
             label='SHASH',
            )
    
    # bnn pit    
    if bnn_cpd is not None:
        bins, hist_bnn, D_bnn, EDp_bnn = model_diagnostics.compute_pit('bnn',onehot_val, bnn_cpd)
        plt.bar(hist_bnn[1][:-1]+3*bins_inc/6+bins_inc/6,
                 hist_bnn[0],
                 width=bins_inc/3,
                 color=clr_bnn,
                 label='BNN',
                )
        plt.text(0.,np.max(yticks)*.94,
                 'BNN ~~~~D: ' + str(np.round(D_bnn,4)) + ' (' + str(np.round(EDp_bnn,3)) +  ')', 
                 color=clr_bnn,
                 verticalalignment='top',             
                 fontsize=12)
        

    # make the figure pretty
    plt.axhline(y=.1, 
                linestyle='--',
                color='dimgray', 
                linewidth=1.,
               )
    plt.ylim(0,.2)
    plt.xticks(bins,np.around(bins,1))
    ax = plt.gca()
    yticks = np.around(np.arange(0,.25,.05),2)
    plt.yticks(yticks,yticks)
    
    plt.text(0.,np.max(yticks)*.99,
             'SHASH D: ' + str(np.round(D_shash,4)) + ' (' + str(np.round(EDp_shash,3)) +  ')', 
             color=clr_shash,     
             verticalalignment='top',
             fontsize=12)


    plt.xlabel('probability integral transform')
    plt.ylabel('probability')
    plt.legend(loc=1)
    plt.title('PIT histogram comparison', fontsize=FS, color='k')
    
def plot_medians(ax, onehot_val, shash_cpd, bnn_cpd, shash_med):
    plt.sca(ax)
    
    bnn_med = np.median(bnn_cpd,axis=1)
    shash_error = np.mean(np.abs(shash_med - onehot_val[:,0]))
    bnn_error = np.mean(np.abs(bnn_med - onehot_val[:,0]))

    plt.plot(shash_med,
             bnn_med,
             linestyle='None',
             marker='.',
             color='dimgray',
             markerfacecolor='violet',
             markersize=7,
             markeredgewidth=.5,
            )

    plt.xlabel('SHASH median')
    plt.ylabel('BNN median')

    plt.plot((-100,100),(-100,100),'--',color='dimgray', linewidth=1.)
    plt.axis('scaled')
    plt.xlim(-25,25)
    plt.ylim(-25,25)

    ax = plt.gca()
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    plt.xticks(xticks.astype(int),xticks.astype(int))
    plt.yticks(yticks.astype(int),yticks.astype(int))

    plt.text(-29,27,
             'SHASH mean $|$error$|$: ' + str(np.round(shash_error,2)), 
             color='dimgray',         
             fontsize=12)
    plt.text(-29,24,
             'BNN ~~~~mean $|$error$|$: ' + str(np.round(bnn_error,2)), 
             color='dimgray',
             fontsize=12)

    plt.title('Median vs Median')    
    
def plot_nlls(ax, x_val, onehot_val, model_shash, shash_cpd, bnn_cpd):
    plt.sca(ax)
    
    shash_nloglike = model_diagnostics.compute_nll('shash', onehot_val, model_shash=model_shash, x_val=x_val)
    bnn_nloglike   = model_diagnostics.compute_nll('bnn', onehot_val, bnn_cpd=bnn_cpd)
    plt.plot(shash_nloglike.numpy(),
             bnn_nloglike,
             linestyle='None',
             marker='.',
             color='dimgray',
             markerfacecolor='violet',
             markersize=7,
             markeredgewidth=.5,
            )

    plt.xlabel('SHASH negative log-likelihood')
    plt.ylabel('BNN negative log-likelihood')

    plt.plot((-100,100),(-100,100),'--',color='dimgray', linewidth=1.)
    plt.axis('scaled')
    plt.xlim(2,11)
    plt.ylim(2,11)

    ax = plt.gca()
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    plt.xticks(xticks.astype(int),xticks.astype(int))
    plt.yticks(yticks.astype(int),yticks.astype(int))

    plt.text(2.1,10.6,
             'SHASH mean NLL: ' + str(np.round(np.nanmean(shash_nloglike.numpy()),3)), 
             color='dimgray',         
             fontsize=12)
    plt.text(2.1,10.2,
             'BNN ~~~~mean NLL: ' + str(np.round(np.nanmean(bnn_nloglike),3)),
             color='dimgray',
             fontsize=12)

    plt.title('NLL Comparison')    
        