from pyscf import gto, scf

# H atom at STO-3G
mol_h = gto.M(atom='H 0 0 0', basis='sto-3g', spin=1)
mf_h  = scf.UHF(mol_h).run()
print(f"H atom STO-3G: {mf_h.e_tot} Hartree")

# F atom at STO-3G  
mol_f = gto.M(atom='F 0 0 0', basis='sto-3g', spin=1)
mf_f  = scf.UHF(mol_f).run()
print(f"F atom STO-3G: {mf_f.e_tot} Hartree")