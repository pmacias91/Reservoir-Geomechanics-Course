import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Homework1():
    # Initializing the variables for both wells
    def __init__(self, barnettData, gomData) -> None:
        self.barnettDepth = barnettData[:,0]
        self.barnettRho = barnettData[:, 1]
        self.gomDepth = gomData[:, 0]
        self.gomRho = gomData[:, 1]
    
    def solve(self):
        "TASK 1"
        
        "BARNETT"
        # EXTRAPOLATION FROM Z = 0 TO FIRST RECORD (Z = 100.5 ft)
        barnettSurfaceDepth = np.linspace(0,min(self.barnettDepth),50)
        barnettSurfaceRho = np.full(50, 1.8778) # create numpy array of constant density values 1.8778 g/cc or barnettRho[0] (first recorded depth in Barnett Shale)
        barnettDepthFinal = np.append([barnettSurfaceDepth], self.barnettDepth)
        barnettRhoFinal = np.append([barnettSurfaceRho], self.barnettRho)

        "GOM"
        # EXTRAPOLATION FROM Z = 0 TO SEAFLOOR (Z = 1000ft)
        gomSeaDepth = np.linspace(0,1000, 11) #values from z=0 to sea floor (depth of 1000ft)
        gomSeaRho = np.full(11, 1) #according to hw1, use a density of 1.0 g/cm3 from the surface to the sea floor (depth of 1000ft)

        # EXTRAPOLATE FROM Z = 1000ft TO FIRST RECORD (Z = 3515 ft)
        # to extrapolate I am going to use the equation of the slope (m) in a straight line m=(y2-y1)/(x2-x1)
        # First I calculate m with the information provided
        delta_y2 = depth2 = min(self.gomDepth); delta_y1 = depth1 = 1000
        delta_x2 = rho2 = 1.7; delta_x1 = rho1 = 1
        m = (delta_y2-delta_y1)/(delta_x2-delta_x1)

        gomSeafloorDepth = np.linspace(1000,min(self.gomDepth),11) # Now I obtain the values for y or depth after the seafloor (depth of 1000ft)
        gomSeafloorRho = ((gomSeafloorDepth - delta_y1)/m) + delta_x1

        # add all the data together for each depth and rho in GOM data
        gomDepthFinal = np.append([gomSeaDepth, gomSeafloorDepth], self.gomDepth)
        gomRhoFinal = np.append([gomSeaRho, gomSeafloorRho], self.gomRho)

        # Create the Figure for the plots
        task1Fig, (barnett_plot, gom_plot) = plt.subplots(1, 2, figsize=(12, 10))

        # Plot Barnett
        barnett_plot.plot(barnettRhoFinal, barnettDepthFinal, 'r')
        barnett_plot.set_title("Barnett Density Log", pad=10, size=10)
        barnett_plot.grid(True)
        barnett_plot.set_ylim(0, (barnettDepthFinal[-1]+500)); barnett_plot.set_xlim(0,5)
        barnett_plot.invert_yaxis()
        barnett_plot.set_xlabel('Density (g/cc)'); barnett_plot.set_ylabel('Depth (ft)')
        barnett_plot.xaxis.set_label_position('top') 
        barnett_plot.xaxis.tick_top()

        # Plot GOM
        gom_plot.plot(gomRhoFinal, gomDepthFinal, 'b' )
        gom_plot.set_title("Barnett Density Log", pad=10, size=10)
        gom_plot.grid(True)
        gom_plot.set_ylim(0, (gomDepthFinal[-1]+500)); gom_plot.set_xlim(0,5)
        gom_plot.invert_yaxis()
        gom_plot.set_xlabel('Density (g/cc)'); gom_plot.set_ylabel('Depth (ft)')
        gom_plot.xaxis.set_label_position('top') 
        gom_plot.xaxis.tick_top()

        plt.tight_layout(pad=2.0) #sets distance between two plots

        "TASK 2"

        "BARNETT"
        #creating the blocks
        valueLimit1a = 480; valueLimit2a = 880; valueLimit3a = 4900; valueLimit4a = 5930
        x1_barnett = np.linspace(0, 5, 5); y1_barnett = np.full(5, valueLimit1a)
        x2_barnett = np.linspace(0, 5, 5); y2_barnett = np.full(5, valueLimit2a)
        x3_barnett = np.linspace(0, 5, 5); y3_barnett = np.full(5, valueLimit3a)
        x4_barnett = np.linspace(0, 5, 5); y4_barnett = np.full(5, valueLimit4a)

        indexLimit1a = np.where(barnettDepthFinal == valueLimit1a)[0][0]
        y_block1_barnett = barnettDepthFinal[:indexLimit1a+1]
        x_block1_barnett = np.full((len(y_block1_barnett)), np.nanmean(barnettRhoFinal[:indexLimit1a+1]))
        
        indexLimit2a = np.where(barnettDepthFinal == valueLimit2a)[0][0]
        y_block2_barnett = barnettDepthFinal[indexLimit1a:indexLimit2a+1]
        x_block2_barnett = np.full((len(y_block2_barnett)), np.nanmean(barnettRhoFinal[indexLimit1a:indexLimit2a+1]))

        indexLimit3a = np.where(barnettDepthFinal == valueLimit3a)[0][0]
        y_block3_barnett = barnettDepthFinal[indexLimit2a:indexLimit3a+1]
        x_block3_barnett = np.full((len(y_block3_barnett)), np.nanmean(barnettRhoFinal[indexLimit2a:indexLimit3a+1]))

        indexLimit4a = np.where(barnettDepthFinal == valueLimit4a)[0][0]
        y_block4_barnett = barnettDepthFinal[indexLimit3a:indexLimit4a+1]
        x_block4_barnett = np.full((len(y_block4_barnett)), np.nanmean(barnettRhoFinal[indexLimit3a:indexLimit4a+1]))

        indexLimit5a = len(barnettDepthFinal)-1
        y_block5_barnett = barnettDepthFinal[indexLimit4a:indexLimit5a+1]
        x_block5_barnett = np.full((len(y_block5_barnett)), np.nanmean(barnettRhoFinal[indexLimit4a:indexLimit5a+1]))

        barnettRhoBlock = np.concatenate([x_block1_barnett, x_block2_barnett, x_block3_barnett, x_block4_barnett, x_block5_barnett])
        barnettDepthBlock = np.concatenate([y_block1_barnett, y_block2_barnett, y_block3_barnett, y_block4_barnett, y_block5_barnett])

        # plot the division lines for the blocks
        barnett_plot.plot(x1_barnett, y1_barnett); barnett_plot.plot(x2_barnett, y2_barnett)
        barnett_plot.plot(x3_barnett, y3_barnett); barnett_plot.plot(x4_barnett, y4_barnett)

        # Plot the average density for each block
        barnett_plot.plot(barnettRhoBlock, barnettDepthBlock, '--',color='black')

        "GOM"
        #creating the blocks
        valueLimit1b = 1000; valueLimit2b = 3515; valueLimit3b = 7840; valueLimit4b = 11070
        
        x1_gom = np.linspace(0, 5, 5); y1_gom = np.full(5, valueLimit1b)
        x2_gom = np.linspace(0, 5, 5); y2_gom = np.full(5, valueLimit2b)
        x3_gom = np.linspace(0, 5, 5); y3_gom = np.full(5, valueLimit3b)
        x4_gom = np.linspace(0, 5, 5); y4_gom = np.full(5, valueLimit4b)

        indexLimit1b = np.where(gomDepthFinal == valueLimit1b)[0][0]
        y_block1_gom = gomDepthFinal[:indexLimit1b+1]
        x_block1_gom = np.full((len(y_block1_gom)), np.nanmean(gomRhoFinal[:indexLimit1b+1]))

        indexLimit2b = np.where(gomDepthFinal == valueLimit2b)[0][0]
        y_block2_gom = gomDepthFinal[indexLimit1b:indexLimit2b+1]
        x_block2_gom = np.full((len(y_block2_gom)), np.nanmean(gomRhoFinal[indexLimit1b:indexLimit2b+1]))

        indexLimit3b = np.where(gomDepthFinal == valueLimit3b)[0][0]
        y_block3_gom = gomDepthFinal[indexLimit2b:indexLimit3b+1]
        x_block3_gom = np.full((len(y_block3_gom)), np.nanmean(gomRhoFinal[indexLimit2b:indexLimit3b+1]))

        indexLimit4b = np.where(gomDepthFinal == valueLimit4b)[0][0]
        y_block4_gom = gomDepthFinal[indexLimit3b:indexLimit4b+1]
        x_block4_gom = np.full((len(y_block4_gom)), np.nanmean(gomRhoFinal[indexLimit3b:indexLimit4b+1]))

        indexLimit5b = len(gomDepthFinal)-1
        y_block5_gom = gomDepthFinal[indexLimit4b:indexLimit5b+1]
        x_block5_gom = np.full((len(y_block5_gom)), np.nanmean(gomRhoFinal[indexLimit4b:indexLimit5b+1]))

        gomBlockDepth = np.concatenate([y_block1_gom, y_block2_gom, y_block3_gom, y_block4_gom, y_block5_gom])
        gomBlockRho = np.concatenate([x_block1_gom, x_block2_gom, x_block3_gom, x_block4_gom, x_block5_gom])
        
        # plot the division lines for the blocks
        gom_plot.plot(x1_gom, y1_gom); gom_plot.plot(x2_gom, y2_gom)
        gom_plot.plot(x3_gom, y3_gom); gom_plot.plot(x4_gom, y4_gom)

        # Plot the average density for each block
        gom_plot.plot(gomBlockRho, gomBlockDepth, '--',color='black')

        "TASK 3"

        "BARNETT"
        # CALCULATE DELTA Z FOR CONTINUOS
        thickness_barnett = np.array([j-i for i, j in zip(barnettDepthFinal[:-1], barnettDepthFinal[1:])])
        thickness_barnett = np.append([(min(barnettDepthFinal) - 0)], thickness_barnett)

        # CALCULATE DELTA Z FOR BLOCK
        thickness_block_barnett = np.array([j-i for i, j in zip(barnettDepthBlock[:-1], barnettDepthBlock[1:])])
        thickness_block_barnett = np.append([(min(barnettDepthBlock) - 0)], thickness_block_barnett)

        # CALCULATE SV FOR CONTINUOS LOG
        overburdenStress_barnet = thickness_barnett * barnettRhoFinal * 8.35 * 0.052
        overburdenStress_barnet = np.cumsum(overburdenStress_barnet)

        # CALCULATE SV FOR BLOCK LOG
        sv_block_barnett = thickness_block_barnett * barnettRhoBlock * 8.35 * 0.052
        sv_block_barnett = np.cumsum(sv_block_barnett)

        "GOM"
        # CALCULATE DELTA Z FOR CONTINUOS
        thickness_gom = np.array([j-i for i, j in zip(gomDepthFinal[:-1], gomDepthFinal[1:])])
        thickness_gom = np.append([(min(gomDepthFinal) - 0)], thickness_gom)

        # CALCULATE DELTA Z FOR BLOCK
        thickness_block_gom = np.array([j-i for i, j in zip(gomBlockDepth[:-1], gomBlockDepth[1:])])
        thickness__block_gom = np.append([(min(gomBlockDepth) - 0)], thickness_block_gom)

        # CALCULATE SV FOR CONTINUOS LOG
        overburdenStress_gom = thickness_gom * gomRhoFinal * 8.35 * 0.052
        overburdenStress_gom = np.cumsum(overburdenStress_gom)

        # CALCULATE SV FOR BLOCK LOG
        sv_block_gom = thickness__block_gom * gomBlockRho * 8.35 * 0.052
        sv_block_gom = np.cumsum(sv_block_gom)

        # CALCULATE HYDROSTATIC PRESSURE
        pp_barnett = 0.44 * barnettDepthFinal
        pp_gom = 0.44 * gomDepthFinal

        task3Fig, (plot_sv_barnett, plot_sv_gom) = plt.subplots(1, 2, figsize=(10, 10))

        "PLOT BARNETT"
        plot_sv_barnett.plot(overburdenStress_barnet, barnettDepthFinal, color='red')
        plot_sv_barnett.plot(sv_block_barnett, barnettDepthBlock, '--', color='blue')
        plot_sv_barnett.plot(pp_barnett, barnettDepthFinal, color='green')
        plot_sv_barnett.set_title("Barnett Stress", pad=10, size=10)
        plot_sv_barnett.legend(['Sv (Continuos log)', 'Sv (Block log)', 'Hydrostatic Pore Pressure'])
        plot_sv_barnett.grid(True)
        plot_sv_barnett.set_ylim(0, (barnettDepthFinal[-1]+500)) 
        plot_sv_barnett.invert_yaxis()
        plot_sv_barnett.set_xlabel('Sv (psi)'); plot_sv_barnett.set_ylabel('Depth (ft)')
        plot_sv_barnett.xaxis.set_label_position('top') 
        plot_sv_barnett.xaxis.tick_top()

        "PLOT GOM"
        plot_sv_gom.plot(overburdenStress_gom, gomDepthFinal, color='red')
        plot_sv_gom.plot(sv_block_gom, gomBlockDepth, '--', color='blue')
        plot_sv_gom.plot(pp_gom, gomDepthFinal, color='green')
        plot_sv_gom.set_title("GOM Stress", pad=10, size=10)
        plot_sv_gom.legend(['Sv (Continuos log)', 'Sv (Block log)', 'Hydrostatic Pore Pressure'])
        plot_sv_gom.grid(True)
        plot_sv_gom.set_ylim(0, (gomDepthFinal[-1]+500)); 
        plot_sv_gom.invert_yaxis()
        plot_sv_gom.set_xlabel('Sv (psi)'); plot_sv_gom.set_ylabel('Depth (ft)')
        plot_sv_gom.xaxis.set_label_position('top') 
        plot_sv_gom.xaxis.tick_top()
        
        
        plt.tight_layout(pad=2.0) #sets distance between two plots

        "TASK 4"

        np.seterr(divide='ignore', invalid='ignore') #ignore the error "RuntimeWarning: invalid value encountered in divide overburdenGradient_barnett = np.divide(overburdenStress_barnet, barnettDepthFinal)"
        "BARNET"        
        overburdenGradient_barnett = np.divide(overburdenStress_barnet, barnettDepthFinal)
        
        task4Fig, (barnett, gom) = plt.subplots(1, 2, figsize=(10, 10), clear=True)
        "PLOT BARNETT"
        barnett.plot(overburdenGradient_barnett, barnettDepthFinal, color='red')
        barnett.set_title("Barnett Gradient Stress", pad=10, size=15)
        barnett.grid(True)
        barnett.set_ylim(0, (barnettDepthFinal[-1]+500))
        barnett.invert_yaxis()
        barnett.set_xlabel('Sv gradient (psi/ft)'); barnett.set_ylabel('Depth (ft)')
        barnett.xaxis.set_label_position('top') 
        barnett.xaxis.tick_top()

        "GOM"
        overburdenGradient_gom = np.divide(overburdenStress_gom, gomDepthFinal)

        "PLOT BARNETT"
        gom.plot(overburdenGradient_gom, gomDepthFinal, color='blue')
        gom.set_title("GOM Gradient Stress", pad=10, size=15)
        gom.grid(True)
        gom.set_ylim(0, (gomDepthFinal[-1]+500))
        gom.invert_yaxis()
        gom.set_xlabel('Sv gradient (psi/ft)'); gom.set_ylabel('Depth (ft)')
        gom.xaxis.set_label_position('top') 
        gom.xaxis.tick_top()

        "EXERCISE 2"
        "BARTNETT"
        rho_matrix_barnett = 2.7
        rho_fluid_barnett = 1
        porosity_barnett = (barnettRhoFinal - rho_matrix_barnett) / (rho_fluid_barnett - rho_matrix_barnett)

        porosityFig, (barnett, gom) = plt.subplots(1, 2, figsize=(10, 10), clear=True)
        # PLOT BARNETT
        barnett.plot(porosity_barnett, barnettDepthFinal, color='red')
        barnett.set_title("Barnett Porosity", pad=10, size=15)
        barnett.grid(True)
        barnett.set_ylim(0, (barnettDepthFinal[-1]+500)); barnett.set_xlim(0, 1)
        barnett.invert_yaxis()
        barnett.set_xlabel('Porotity'); barnett.set_ylabel('Depth (ft)')
        barnett.xaxis.set_label_position('top') 
        barnett.xaxis.tick_top()

        "GOM"
        rho_matrix_gom = 2.7
        rho_fluid_gom = 1
        porosity_gom = (gomRhoFinal - rho_matrix_gom) / (rho_fluid_gom - rho_matrix_gom)

        # PLOT BARNETT
        gom.plot(porosity_gom, gomDepthFinal, color='blue')
        gom.set_title("GOM Porosity", pad=10, size=15)
        gom.grid(True)
        gom.set_ylim(0, (gomDepthFinal[-1]+500)); gom.set_xlim(0, 1)
        gom.invert_yaxis()
        gom.set_xlabel('Porotity'); gom.set_ylabel('Depth (ft)')
        gom.xaxis.set_label_position('top') 
        gom.xaxis.tick_top()

        task1Fig.suptitle('HOMEWORK 1 - TASK 1 & 2', fontsize=15)

        plt.show()

# Read and load the data from the .txt files
barnettData = np.loadtxt('Homework1/Barnett_density_data.txt',skiprows=2)
gomData = np.loadtxt('Homework1/GOM_offshore_density_data.txt',skiprows=2)

Homework1(barnettData,gomData).solve()