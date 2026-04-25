cisd_eqs_ovlp = [
    [1.0, '->', ['', ''], ['T', 'X']],
    [1.0, 'ji,ij->', ['ii', 'ii', ''], ['T', 'A', 'X']],
    [1.0, 'ji,ij->', ['II', 'II', ''], ['T', 'B', 'X']],
    [0.5, 'jlik,ij,kl->', ['iiii', 'ii', 'ii', ''], ['T', 'A', 'A', 'X']],
    [1.0, 'jlik,ij,kl->', ['iIiI', 'ii', 'II', ''], ['T', 'A', 'B', 'X']],
    [0.5, 'jlik,ij,kl->', ['IIII', 'II', 'II', ''], ['T', 'B', 'B', 'X']],
]

cisdt_eqs_ovlp = [
    [1.0, '->', ['', ''], ['T', 'X']],
    [1.0, 'ji,ij->', ['ii', 'ii', ''], ['T', 'A', 'X']],
    [1.0, 'ji,ij->', ['II', 'II', ''], ['T', 'B', 'X']],
    [0.5, 'jlik,ij,kl->', ['iiii', 'ii', 'ii', ''], ['T', 'A', 'A', 'X']],
    [1.0, 'jlik,ij,kl->', ['iIiI', 'ii', 'II', ''], ['T', 'A', 'B', 'X']],
    [0.5, 'jlik,ij,kl->', ['IIII', 'II', 'II', ''], ['T', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnikm,ij,kl,mn->', ['iiiiii', 'ii', 'ii', 'ii', ''], ['T', 'A', 'A', 'A', 'X']],
    [0.5, 'jlnikm,ij,kl,mn->', ['iiIiiI', 'ii', 'ii', 'II', ''], ['T', 'A', 'A', 'B', 'X']],
    [0.5, 'jlnikm,ij,kl,mn->', ['iIIiII', 'ii', 'II', 'II', ''], ['T', 'A', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnikm,ij,kl,mn->', ['IIIIII', 'II', 'II', 'II', ''], ['T', 'B', 'B', 'B', 'X']],
]

cisdtq_eqs_ovlp = [
    [1.0, '->', ['', ''], ['T', 'X']],
    [1.0, 'ji,ij->', ['ii', 'ii', ''], ['T', 'A', 'X']],
    [1.0, 'ji,ij->', ['II', 'II', ''], ['T', 'B', 'X']],
    [0.5, 'jlik,ij,kl->', ['iiii', 'ii', 'ii', ''], ['T', 'A', 'A', 'X']],
    [1.0, 'jlik,ij,kl->', ['iIiI', 'ii', 'II', ''], ['T', 'A', 'B', 'X']],
    [0.5, 'jlik,ij,kl->', ['IIII', 'II', 'II', ''], ['T', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnikm,ij,kl,mn->', ['iiiiii', 'ii', 'ii', 'ii', ''], ['T', 'A', 'A', 'A', 'X']],
    [0.5, 'jlnikm,ij,kl,mn->', ['iiIiiI', 'ii', 'ii', 'II', ''], ['T', 'A', 'A', 'B', 'X']],
    [0.5, 'jlnikm,ij,kl,mn->', ['iIIiII', 'ii', 'II', 'II', ''], ['T', 'A', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnikm,ij,kl,mn->', ['IIIIII', 'II', 'II', 'II', ''], ['T', 'B', 'B', 'B', 'X']],
    [0.041666666666666664, 'jlnpikmo,ij,kl,mn,op->', ['iiiiiiii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'A', 'A', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnpikmo,ij,kl,mn,op->', ['iiiIiiiI', 'ii', 'ii', 'ii', 'II', ''], ['T', 'A', 'A', 'A', 'B', 'X']],
    [0.25, 'jlnpikmo,ij,kl,mn,op->', ['iiIIiiII', 'ii', 'ii', 'II', 'II', ''], ['T', 'A', 'A', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnpikmo,ij,kl,mn,op->', ['iIIIiIII', 'ii', 'II', 'II', 'II', ''], ['T', 'A', 'B', 'B', 'B', 'X']],
    [0.041666666666666664, 'jlnpikmo,ij,kl,mn,op->', ['IIIIIIII', 'II', 'II', 'II', 'II', ''], ['T', 'B', 'B', 'B', 'B', 'X']],
]

cisd_eqs = [
    [1.0, ',ji,ij->', ['', 'ii', 'ii', ''], ['T', 'H', 'A', 'X']],
    [1.0, ',ji,ij->', ['', 'II', 'II', ''], ['T', 'H', 'B', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['II', 'ii', 'II', 'ii', ''], ['T', 'H', 'B', 'A', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'X']],
    [1.0, 'ja,bi,ab,ij->', ['ie', 'ei', 'ee', 'ii', ''], ['H', 'T', 'A', 'A', 'X']],
    [0.5, ',jlik,ij,kl->', ['', 'iiii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'X']],
    [1.0, ',jlik,ij,kl->', ['', 'iIiI', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'X']],
    [1.0, 'ja,bi,ab,ij->', ['IE', 'EI', 'EE', 'II', ''], ['H', 'T', 'B', 'B', 'X']],
    [0.5, ',jlik,ij,kl->', ['', 'IIII', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['iiii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['iiii', 'II', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'X']],
    [1.0, 'jlik,nm,ij,kl,mn->', ['iIiI', 'ii', 'ii', 'II', 'ii', ''], ['T', 'H', 'A', 'B', 'A', 'X']],
    [1.0, 'jlik,nm,ij,kl,mn->', ['iIiI', 'II', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['iiie', 'ei', 'ii', 'ee', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['ii', 'iiii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'X']],
    [1.0, 'ji,lnkm,ij,kl,mn->', ['ii', 'iIiI', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['iIiE', 'EI', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['ii', 'IIII', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['IIII', 'ii', 'II', 'II', 'ii', ''], ['T', 'H', 'B', 'B', 'A', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['IIII', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'X']],
    [1.0, 'ljai,bk,ij,ab,kl->', ['iIeI', 'ei', 'II', 'ee', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['II', 'iiii', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'A', 'A', 'X']],
    [1.0, 'ji,lnkm,ij,kl,mn->', ['II', 'iIiI', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'A', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['IIIE', 'EI', 'II', 'EE', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['II', 'IIII', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['ie', 'eiii', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['ie', 'eIiI', 'ee', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'X']],
    [1.0, 'la,jbik,ij,ab,kl->', ['IE', 'iEiI', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['IE', 'EIII', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'iiii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'iIiI', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'IIII', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'iiii', 'ii', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'B', 'A', 'A', 'X']],
    [1.0, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'iIiI', 'ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'A', 'B', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'IIII', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iiie', 'eiii', 'ii', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iiie', 'eIiI', 'ii', 'ee', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'X']],
    [1.0, 'jnia,lbkm,ij,kl,ab,mn->', ['iIiE', 'iEiI', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iIiE', 'EIII', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'iiii', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'B', 'A', 'A', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'iIiI', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'B', 'A', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'IIII', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'X']],
    [1.0, 'ljai,bnkm,ij,ab,kl,mn->', ['iIeI', 'eiii', 'II', 'ee', 'ii', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'X']],
    [1.0, 'ljai,bnkm,ij,ab,kl,mn->', ['iIeI', 'eIiI', 'II', 'ee', 'ii', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'X']],
    [1.0, 'jnia,lbkm,ij,kl,ab,mn->', ['IIIE', 'iEiI', 'II', 'ii', 'EE', 'II', ''], ['H', 'T', 'B', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['IIIE', 'EIII', 'II', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdik,ab,cd,ij,kl->', ['iiee', 'eeii', 'ee', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'jlac,bdik,ab,ij,cd,kl->', ['iIeE', 'eEiI', 'ee', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [0.25, 'jlac,bdik,ab,cd,ij,kl->', ['IIEE', 'EEII', 'EE', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'X']],
]

cisdt_eqs = [
    [1.0, ',ji,ij->', ['', 'ii', 'ii', ''], ['T', 'H', 'A', 'X']],
    [1.0, ',ji,ij->', ['', 'II', 'II', ''], ['T', 'H', 'B', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['II', 'ii', 'II', 'ii', ''], ['T', 'H', 'B', 'A', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'X']],
    [1.0, 'ja,bi,ab,ij->', ['ie', 'ei', 'ee', 'ii', ''], ['H', 'T', 'A', 'A', 'X']],
    [0.5, ',jlik,ij,kl->', ['', 'iiii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'X']],
    [1.0, ',jlik,ij,kl->', ['', 'iIiI', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'X']],
    [1.0, 'ja,bi,ab,ij->', ['IE', 'EI', 'EE', 'II', ''], ['H', 'T', 'B', 'B', 'X']],
    [0.5, ',jlik,ij,kl->', ['', 'IIII', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['iiii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['iiii', 'II', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'X']],
    [1.0, 'jlik,nm,ij,kl,mn->', ['iIiI', 'ii', 'ii', 'II', 'ii', ''], ['T', 'H', 'A', 'B', 'A', 'X']],
    [1.0, 'jlik,nm,ij,kl,mn->', ['iIiI', 'II', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['iiie', 'ei', 'ii', 'ee', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['ii', 'iiii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'X']],
    [1.0, 'ji,lnkm,ij,kl,mn->', ['ii', 'iIiI', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['iIiE', 'EI', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['ii', 'IIII', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['IIII', 'ii', 'II', 'II', 'ii', ''], ['T', 'H', 'B', 'B', 'A', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['IIII', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'X']],
    [1.0, 'ljai,bk,ij,ab,kl->', ['iIeI', 'ei', 'II', 'ee', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['II', 'iiii', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'A', 'A', 'X']],
    [1.0, 'ji,lnkm,ij,kl,mn->', ['II', 'iIiI', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'A', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['IIIE', 'EI', 'II', 'EE', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['II', 'IIII', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['ie', 'eiii', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['ie', 'eIiI', 'ee', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'X']],
    [1.0, 'la,jbik,ij,ab,kl->', ['IE', 'iEiI', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['IE', 'EIII', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnikm,po,ij,kl,mn,op->', ['iiiiii', 'ii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnikm,po,ij,kl,mn,op->', ['iiiiii', 'II', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'X']],
    [0.5, 'jlnikm,po,ij,kl,mn,op->', ['iiIiiI', 'ii', 'ii', 'ii', 'II', 'ii', ''], ['T', 'H', 'A', 'A', 'B', 'A', 'X']],
    [0.5, 'jlnikm,po,ij,kl,mn,op->', ['iiIiiI', 'II', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'iiii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'iIiI', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'IIII', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jlnikm,po,ij,kl,mn,op->', ['iIIiII', 'ii', 'ii', 'II', 'II', 'ii', ''], ['T', 'H', 'A', 'B', 'B', 'A', 'X']],
    [0.5, 'jlnikm,po,ij,kl,mn,op->', ['iIIiII', 'II', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'iiii', 'ii', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'B', 'A', 'A', 'X']],
    [1.0, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'iIiI', 'ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'A', 'B', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'IIII', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iiie', 'eiii', 'ii', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iiie', 'eIiI', 'ii', 'ee', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'X']],
    [1.0, 'jnia,lbkm,ij,kl,ab,mn->', ['iIiE', 'iEiI', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iIiE', 'EIII', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnikm,po,ij,kl,mn,op->', ['IIIIII', 'ii', 'II', 'II', 'II', 'ii', ''], ['T', 'H', 'B', 'B', 'B', 'A', 'X']],
    [0.16666666666666666, 'jlnikm,po,ij,kl,mn,op->', ['IIIIII', 'II', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'iiii', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'B', 'A', 'A', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'iIiI', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'B', 'A', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'IIII', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'X']],
    [1.0, 'ljai,bnkm,ij,ab,kl,mn->', ['iIeI', 'eiii', 'II', 'ee', 'ii', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'X']],
    [1.0, 'ljai,bnkm,ij,ab,kl,mn->', ['iIeI', 'eIiI', 'II', 'ee', 'ii', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'X']],
    [1.0, 'jnia,lbkm,ij,kl,ab,mn->', ['IIIE', 'iEiI', 'II', 'ii', 'EE', 'II', ''], ['H', 'T', 'B', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['IIIE', 'EIII', 'II', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdik,ab,cd,ij,kl->', ['iiee', 'eeii', 'ee', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'X']],
    [0.5, 'ja,blnikm,ab,ij,kl,mn->', ['ie', 'eiiiii', 'ee', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'ja,blnikm,ab,ij,kl,mn->', ['ie', 'eiIiiI', 'ee', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'X']],
    [1.0, 'jlac,bdik,ab,ij,cd,kl->', ['iIeE', 'eEiI', 'ee', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'ja,blnikm,ab,ij,kl,mn->', ['ie', 'eIIiII', 'ee', 'ii', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'na,jlbikm,ij,kl,ab,mn->', ['IE', 'iiEiiI', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'la,jbnikm,ij,ab,kl,mn->', ['IE', 'iEIiII', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdik,ab,cd,ij,kl->', ['IIEE', 'EEII', 'EE', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'X']],
    [0.5, 'ja,blnikm,ab,ij,kl,mn->', ['IE', 'EIIIII', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'X']],
    [0.08333333333333333, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiiiii', 'iiii', 'ii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiiiii', 'iIiI', 'ii', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.08333333333333333, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiiiii', 'IIII', 'ii', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.25, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiIiiI', 'iiii', 'ii', 'ii', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'B', 'A', 'A', 'X']],
    [0.5, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiIiiI', 'iIiI', 'ii', 'ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'A', 'B', 'X']],
    [0.25, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiIiiI', 'IIII', 'ii', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.25, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iIIiII', 'iiii', 'ii', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'B', 'B', 'A', 'A', 'X']],
    [0.5, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iIIiII', 'iIiI', 'ii', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'A', 'B', 'X']],
    [0.25, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iIIiII', 'IIII', 'ii', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.5, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['iiie', 'eiiiii', 'ii', 'ee', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['iiie', 'eiIiiI', 'ii', 'ee', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.5, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['iiie', 'eIIiII', 'ii', 'ee', 'ii', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jpia,lnbkmo,ij,kl,mn,ab,op->', ['iIiE', 'iiEiiI', 'ii', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jnia,lbpkmo,ij,kl,ab,mn,op->', ['iIiE', 'iEIiII', 'ii', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['iIiE', 'EIIIII', 'ii', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.08333333333333333, 'jlnikm,proq,ij,kl,mn,op,qr->', ['IIIIII', 'iiii', 'II', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'B', 'B', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnikm,proq,ij,kl,mn,op,qr->', ['IIIIII', 'iIiI', 'II', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'A', 'B', 'X']],
    [0.08333333333333333, 'jlnikm,proq,ij,kl,mn,op,qr->', ['IIIIII', 'IIII', 'II', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.5, 'ljai,bnpkmo,ij,ab,kl,mn,op->', ['iIeI', 'eiiiii', 'II', 'ee', 'ii', 'ii', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'ljai,bnpkmo,ij,ab,kl,mn,op->', ['iIeI', 'eiIiiI', 'II', 'ee', 'ii', 'ii', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'B', 'X']],
    [0.5, 'ljai,bnpkmo,ij,ab,kl,mn,op->', ['iIeI', 'eIIiII', 'II', 'ee', 'ii', 'II', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jpia,lnbkmo,ij,kl,mn,ab,op->', ['IIIE', 'iiEiiI', 'II', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jnia,lbpkmo,ij,kl,ab,mn,op->', ['IIIE', 'iEIiII', 'II', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['IIIE', 'EIIIII', 'II', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdnikm,ab,cd,ij,kl,mn->', ['iiee', 'eeiiii', 'ee', 'ee', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.25, 'jlac,bdnikm,ab,cd,ij,kl,mn->', ['iiee', 'eeIiiI', 'ee', 'ee', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'X']],
    [1.0, 'jnac,bldikm,ab,ij,kl,cd,mn->', ['iIeE', 'eiEiiI', 'ee', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jlac,bdnikm,ab,ij,cd,kl,mn->', ['iIeE', 'eEIiII', 'ee', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.25, 'lnac,jbdikm,ij,ab,cd,kl,mn->', ['IIEE', 'iEEiII', 'ii', 'EE', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdnikm,ab,cd,ij,kl,mn->', ['IIEE', 'EEIIII', 'EE', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'B', 'X']],
]

cisdtq_eqs = [
    [1.0, ',ji,ij->', ['', 'ii', 'ii', ''], ['T', 'H', 'A', 'X']],
    [1.0, ',ji,ij->', ['', 'II', 'II', ''], ['T', 'H', 'B', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['II', 'ii', 'II', 'ii', ''], ['T', 'H', 'B', 'A', 'X']],
    [1.0, 'ji,lk,ij,kl->', ['II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'X']],
    [1.0, 'ja,bi,ab,ij->', ['ie', 'ei', 'ee', 'ii', ''], ['H', 'T', 'A', 'A', 'X']],
    [0.5, ',jlik,ij,kl->', ['', 'iiii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'X']],
    [1.0, ',jlik,ij,kl->', ['', 'iIiI', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'X']],
    [1.0, 'ja,bi,ab,ij->', ['IE', 'EI', 'EE', 'II', ''], ['H', 'T', 'B', 'B', 'X']],
    [0.5, ',jlik,ij,kl->', ['', 'IIII', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['iiii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['iiii', 'II', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'X']],
    [1.0, 'jlik,nm,ij,kl,mn->', ['iIiI', 'ii', 'ii', 'II', 'ii', ''], ['T', 'H', 'A', 'B', 'A', 'X']],
    [1.0, 'jlik,nm,ij,kl,mn->', ['iIiI', 'II', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['iiie', 'ei', 'ii', 'ee', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['ii', 'iiii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'X']],
    [1.0, 'ji,lnkm,ij,kl,mn->', ['ii', 'iIiI', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['iIiE', 'EI', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['ii', 'IIII', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['IIII', 'ii', 'II', 'II', 'ii', ''], ['T', 'H', 'B', 'B', 'A', 'X']],
    [0.5, 'jlik,nm,ij,kl,mn->', ['IIII', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'X']],
    [1.0, 'ljai,bk,ij,ab,kl->', ['iIeI', 'ei', 'II', 'ee', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['II', 'iiii', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'A', 'A', 'X']],
    [1.0, 'ji,lnkm,ij,kl,mn->', ['II', 'iIiI', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'A', 'B', 'X']],
    [1.0, 'jlia,bk,ij,ab,kl->', ['IIIE', 'EI', 'II', 'EE', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'X']],
    [0.5, 'ji,lnkm,ij,kl,mn->', ['II', 'IIII', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['ie', 'eiii', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['ie', 'eIiI', 'ee', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'X']],
    [1.0, 'la,jbik,ij,ab,kl->', ['IE', 'iEiI', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'X']],
    [1.0, 'ja,blik,ab,ij,kl->', ['IE', 'EIII', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnikm,po,ij,kl,mn,op->', ['iiiiii', 'ii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnikm,po,ij,kl,mn,op->', ['iiiiii', 'II', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'X']],
    [0.5, 'jlnikm,po,ij,kl,mn,op->', ['iiIiiI', 'ii', 'ii', 'ii', 'II', 'ii', ''], ['T', 'H', 'A', 'A', 'B', 'A', 'X']],
    [0.5, 'jlnikm,po,ij,kl,mn,op->', ['iiIiiI', 'II', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'iiii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'iIiI', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['iiii', 'IIII', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jlnikm,po,ij,kl,mn,op->', ['iIIiII', 'ii', 'ii', 'II', 'II', 'ii', ''], ['T', 'H', 'A', 'B', 'B', 'A', 'X']],
    [0.5, 'jlnikm,po,ij,kl,mn,op->', ['iIIiII', 'II', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'iiii', 'ii', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'B', 'A', 'A', 'X']],
    [1.0, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'iIiI', 'ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'A', 'B', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['iIiI', 'IIII', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iiie', 'eiii', 'ii', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iiie', 'eIiI', 'ii', 'ee', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'X']],
    [1.0, 'jnia,lbkm,ij,kl,ab,mn->', ['iIiE', 'iEiI', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['iIiE', 'EIII', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnikm,po,ij,kl,mn,op->', ['IIIIII', 'ii', 'II', 'II', 'II', 'ii', ''], ['T', 'H', 'B', 'B', 'B', 'A', 'X']],
    [0.16666666666666666, 'jlnikm,po,ij,kl,mn,op->', ['IIIIII', 'II', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'iiii', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'B', 'A', 'A', 'X']],
    [0.5, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'iIiI', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'B', 'A', 'B', 'X']],
    [0.25, 'jlik,npmo,ij,kl,mn,op->', ['IIII', 'IIII', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'X']],
    [1.0, 'ljai,bnkm,ij,ab,kl,mn->', ['iIeI', 'eiii', 'II', 'ee', 'ii', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'X']],
    [1.0, 'ljai,bnkm,ij,ab,kl,mn->', ['iIeI', 'eIiI', 'II', 'ee', 'ii', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'X']],
    [1.0, 'jnia,lbkm,ij,kl,ab,mn->', ['IIIE', 'iEiI', 'II', 'ii', 'EE', 'II', ''], ['H', 'T', 'B', 'A', 'B', 'B', 'X']],
    [1.0, 'jlia,bnkm,ij,ab,kl,mn->', ['IIIE', 'EIII', 'II', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdik,ab,cd,ij,kl->', ['iiee', 'eeii', 'ee', 'ee', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'X']],
    [0.5, 'ja,blnikm,ab,ij,kl,mn->', ['ie', 'eiiiii', 'ee', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'ja,blnikm,ab,ij,kl,mn->', ['ie', 'eiIiiI', 'ee', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'X']],
    [1.0, 'jlac,bdik,ab,ij,cd,kl->', ['iIeE', 'eEiI', 'ee', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'ja,blnikm,ab,ij,kl,mn->', ['ie', 'eIIiII', 'ee', 'ii', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'na,jlbikm,ij,kl,ab,mn->', ['IE', 'iiEiiI', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'la,jbnikm,ij,ab,kl,mn->', ['IE', 'iEIiII', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdik,ab,cd,ij,kl->', ['IIEE', 'EEII', 'EE', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'X']],
    [0.5, 'ja,blnikm,ab,ij,kl,mn->', ['IE', 'EIIIII', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'X']],
    [0.041666666666666664, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['iiiiiiii', 'ii', 'ii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.041666666666666664, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['iiiiiiii', 'II', 'ii', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.16666666666666666, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['iiiIiiiI', 'ii', 'ii', 'ii', 'ii', 'II', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'A', 'X']],
    [0.16666666666666666, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['iiiIiiiI', 'II', 'ii', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.08333333333333333, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiiiii', 'iiii', 'ii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiiiii', 'iIiI', 'ii', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.08333333333333333, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiiiii', 'IIII', 'ii', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.25, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['iiIIiiII', 'ii', 'ii', 'ii', 'II', 'II', 'ii', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'A', 'X']],
    [0.25, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['iiIIiiII', 'II', 'ii', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.25, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiIiiI', 'iiii', 'ii', 'ii', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'B', 'A', 'A', 'X']],
    [0.5, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiIiiI', 'iIiI', 'ii', 'ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'A', 'B', 'X']],
    [0.25, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iiIiiI', 'IIII', 'ii', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['iIIIiIII', 'ii', 'ii', 'II', 'II', 'II', 'ii', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'A', 'X']],
    [0.16666666666666666, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['iIIIiIII', 'II', 'ii', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iIIiII', 'iiii', 'ii', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'B', 'B', 'A', 'A', 'X']],
    [0.5, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iIIiII', 'iIiI', 'ii', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'A', 'B', 'X']],
    [0.25, 'jlnikm,proq,ij,kl,mn,op,qr->', ['iIIiII', 'IIII', 'ii', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.5, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['iiie', 'eiiiii', 'ii', 'ee', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['iiie', 'eiIiiI', 'ii', 'ee', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.5, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['iiie', 'eIIiII', 'ii', 'ee', 'ii', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jpia,lnbkmo,ij,kl,mn,ab,op->', ['iIiE', 'iiEiiI', 'ii', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jnia,lbpkmo,ij,kl,ab,mn,op->', ['iIiE', 'iEIiII', 'ii', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['iIiE', 'EIIIII', 'ii', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.041666666666666664, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['IIIIIIII', 'ii', 'II', 'II', 'II', 'II', 'ii', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'A', 'X']],
    [0.041666666666666664, 'jlnpikmo,rq,ij,kl,mn,op,qr->', ['IIIIIIII', 'II', 'II', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.08333333333333333, 'jlnikm,proq,ij,kl,mn,op,qr->', ['IIIIII', 'iiii', 'II', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'B', 'B', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnikm,proq,ij,kl,mn,op,qr->', ['IIIIII', 'iIiI', 'II', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'A', 'B', 'X']],
    [0.08333333333333333, 'jlnikm,proq,ij,kl,mn,op,qr->', ['IIIIII', 'IIII', 'II', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.5, 'ljai,bnpkmo,ij,ab,kl,mn,op->', ['iIeI', 'eiiiii', 'II', 'ee', 'ii', 'ii', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'A', 'X']],
    [1.0, 'ljai,bnpkmo,ij,ab,kl,mn,op->', ['iIeI', 'eiIiiI', 'II', 'ee', 'ii', 'ii', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'B', 'X']],
    [0.5, 'ljai,bnpkmo,ij,ab,kl,mn,op->', ['iIeI', 'eIIiII', 'II', 'ee', 'ii', 'II', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jpia,lnbkmo,ij,kl,mn,ab,op->', ['IIIE', 'iiEiiI', 'II', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jnia,lbpkmo,ij,kl,ab,mn,op->', ['IIIE', 'iEIiII', 'II', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jlia,bnpkmo,ij,ab,kl,mn,op->', ['IIIE', 'EIIIII', 'II', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdnikm,ab,cd,ij,kl,mn->', ['iiee', 'eeiiii', 'ee', 'ee', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.25, 'jlac,bdnikm,ab,cd,ij,kl,mn->', ['iiee', 'eeIiiI', 'ee', 'ee', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.16666666666666666, 'ja,blnpikmo,ab,ij,kl,mn,op->', ['ie', 'eiiiiiii', 'ee', 'ii', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.5, 'ja,blnpikmo,ab,ij,kl,mn,op->', ['ie', 'eiiIiiiI', 'ee', 'ii', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'X']],
    [1.0, 'jnac,bldikm,ab,ij,kl,cd,mn->', ['iIeE', 'eiEiiI', 'ee', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'ja,blnpikmo,ab,ij,kl,mn,op->', ['ie', 'eiIIiiII', 'ee', 'ii', 'ii', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jlac,bdnikm,ab,ij,cd,kl,mn->', ['iIeE', 'eEIiII', 'ee', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'ja,blnpikmo,ab,ij,kl,mn,op->', ['ie', 'eIIIiIII', 'ee', 'ii', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'pa,jlnbikmo,ij,kl,mn,ab,op->', ['IE', 'iiiEiiiI', 'ii', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'na,jlbpikmo,ij,kl,ab,mn,op->', ['IE', 'iiEIiiII', 'ii', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.25, 'lnac,jbdikm,ij,ab,cd,kl,mn->', ['IIEE', 'iEEiII', 'ii', 'EE', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.5, 'la,jbnpikmo,ij,ab,kl,mn,op->', ['IE', 'iEIIiIII', 'ii', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'jlac,bdnikm,ab,cd,ij,kl,mn->', ['IIEE', 'EEIIII', 'EE', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'ja,blnpikmo,ab,ij,kl,mn,op->', ['IE', 'EIIIIIII', 'EE', 'II', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.020833333333333332, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiiiiiii', 'iiii', 'ii', 'ii', 'ii', 'ii', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.041666666666666664, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiiiiiii', 'iIiI', 'ii', 'ii', 'ii', 'ii', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.020833333333333332, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiiiiiii', 'IIII', 'ii', 'ii', 'ii', 'ii', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.08333333333333333, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiiIiiiI', 'iiii', 'ii', 'ii', 'ii', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiiIiiiI', 'iIiI', 'ii', 'ii', 'ii', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'A', 'B', 'X']],
    [0.08333333333333333, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiiIiiiI', 'IIII', 'ii', 'ii', 'ii', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.125, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiIIiiII', 'iiii', 'ii', 'ii', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'A', 'A', 'X']],
    [0.25, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiIIiiII', 'iIiI', 'ii', 'ii', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'A', 'B', 'X']],
    [0.125, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iiIIiiII', 'IIII', 'ii', 'ii', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.08333333333333333, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iIIIiIII', 'iiii', 'ii', 'II', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'A', 'A', 'X']],
    [0.16666666666666666, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iIIIiIII', 'iIiI', 'ii', 'II', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'A', 'B', 'X']],
    [0.08333333333333333, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['iIIIiIII', 'IIII', 'ii', 'II', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'A', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlia,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iiie', 'eiiiiiii', 'ii', 'ee', 'ii', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.5, 'jlia,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iiie', 'eiiIiiiI', 'ii', 'ee', 'ii', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.5, 'jlia,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iiie', 'eiIIiiII', 'ii', 'ee', 'ii', 'ii', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlia,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iiie', 'eIIIiIII', 'ii', 'ee', 'ii', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jria,lnpbkmoq,ij,kl,mn,op,ab,qr->', ['iIiE', 'iiiEiiiI', 'ii', 'ii', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jpia,lnbrkmoq,ij,kl,mn,ab,op,qr->', ['iIiE', 'iiEIiiII', 'ii', 'ii', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jnia,lbprkmoq,ij,kl,ab,mn,op,qr->', ['iIiE', 'iEIIiIII', 'ii', 'ii', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlia,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iIiE', 'EIIIIIII', 'ii', 'EE', 'II', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.020833333333333332, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['IIIIIIII', 'iiii', 'II', 'II', 'II', 'II', 'ii', 'ii', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'A', 'A', 'X']],
    [0.041666666666666664, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['IIIIIIII', 'iIiI', 'II', 'II', 'II', 'II', 'ii', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'A', 'B', 'X']],
    [0.020833333333333332, 'jlnpikmo,rtqs,ij,kl,mn,op,qr,st->', ['IIIIIIII', 'IIII', 'II', 'II', 'II', 'II', 'II', 'II', ''], ['T', 'H', 'B', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'ljai,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iIeI', 'eiiiiiii', 'II', 'ee', 'ii', 'ii', 'ii', 'ii', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.5, 'ljai,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iIeI', 'eiiIiiiI', 'II', 'ee', 'ii', 'ii', 'ii', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.5, 'ljai,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iIeI', 'eiIIiiII', 'II', 'ee', 'ii', 'ii', 'II', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.16666666666666666, 'ljai,bnprkmoq,ij,ab,kl,mn,op,qr->', ['iIeI', 'eIIIiIII', 'II', 'ee', 'ii', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jria,lnpbkmoq,ij,kl,mn,op,ab,qr->', ['IIIE', 'iiiEiiiI', 'II', 'ii', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jpia,lnbrkmoq,ij,kl,mn,ab,op,qr->', ['IIIE', 'iiEIiiII', 'II', 'ii', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'B', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jnia,lbprkmoq,ij,kl,ab,mn,op,qr->', ['IIIE', 'iEIIiIII', 'II', 'ii', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.16666666666666666, 'jlia,bnprkmoq,ij,ab,kl,mn,op,qr->', ['IIIE', 'EIIIIIII', 'II', 'EE', 'II', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.125, 'jlac,bdnpikmo,ab,cd,ij,kl,mn,op->', ['iiee', 'eeiiiiii', 'ee', 'ee', 'ii', 'ii', 'ii', 'ii', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'A', 'X']],
    [0.25, 'jlac,bdnpikmo,ab,cd,ij,kl,mn,op->', ['iiee', 'eeiIiiiI', 'ee', 'ee', 'ii', 'ii', 'ii', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'A', 'B', 'X']],
    [0.125, 'jlac,bdnpikmo,ab,cd,ij,kl,mn,op->', ['iiee', 'eeIIiiII', 'ee', 'ee', 'ii', 'ii', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'B', 'X']],
    [0.5, 'jpac,blndikmo,ab,ij,kl,mn,cd,op->', ['iIeE', 'eiiEiiiI', 'ee', 'ii', 'ii', 'ii', 'EE', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'A', 'B', 'B', 'X']],
    [1.0, 'jnac,bldpikmo,ab,ij,kl,cd,mn,op->', ['iIeE', 'eiEIiiII', 'ee', 'ii', 'ii', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'A', 'B', 'B', 'B', 'X']],
    [0.5, 'jlac,bdnpikmo,ab,ij,cd,kl,mn,op->', ['iIeE', 'eEIIiIII', 'ee', 'ii', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.125, 'npac,jlbdikmo,ij,kl,ab,cd,mn,op->', ['IIEE', 'iiEEiiII', 'ii', 'ii', 'EE', 'EE', 'II', 'II', ''], ['H', 'T', 'A', 'A', 'B', 'B', 'B', 'B', 'X']],
    [0.25, 'lnac,jbdpikmo,ij,ab,cd,kl,mn,op->', ['IIEE', 'iEEIiIII', 'ii', 'EE', 'EE', 'II', 'II', 'II', ''], ['H', 'T', 'A', 'B', 'B', 'B', 'B', 'B', 'X']],
    [0.125, 'jlac,bdnpikmo,ab,cd,ij,kl,mn,op->', ['IIEE', 'EEIIIIII', 'EE', 'EE', 'II', 'II', 'II', 'II', ''], ['H', 'T', 'B', 'B', 'B', 'B', 'B', 'B', 'X']],
]

def lowdin_pair(cw, cx, metric, thrd=1E-20):
    import numpy as np
    swx = cw.T @ metric @ cx
    if np.max(np.abs(swx - np.diag(np.diag(swx)))) > thrd:
        l, _, v = np.linalg.svd(swx)
        cw, cx = cw @ l, cx @ v.T
        cw[:, 0] *= np.linalg.det(l)
        cx[:, 0] *= np.linalg.det(v.T)
        swx = cw.T @ metric @ cx
    assert np.max(np.abs(swx - np.diag(np.diag(swx)))) < 1E-10
    return np.diag(swx), cw, cx

def reduced_overlap(sxx, thrd=1E-8):
    import numpy as np
    sxx_t = np.array(sxx)
    sxx_t[np.abs(sxx) <= thrd] = 1
    reduced_ovlp = np.prod(sxx[np.abs(sxx) > thrd])
    zeros = np.mgrid[:len(sxx)][np.abs(sxx) <= thrd]
    assert len(zeros) == 0
    return 1 / sxx_t, reduced_ovlp, zeros

def ut_ao2mo(mf):
    from pyscf import ao2mo
    import numpy as np

    mol = mf.mol
    ncore = 0
    mo_a, mo_b = mf.mo_coeff
    ncas = mo_a.shape[1] - ncore

    nocca = (mol.nelectron + mol.spin) // 2
    noccb = (mol.nelectron - mol.spin) // 2
    nvira, nvirb = ncas - nocca, ncas - noccb

    mo_core = mo_a[:, :ncore], mo_b[:, :ncore]
    mo_cas = mo_a[:, ncore : ncore + ncas], mo_b[:, ncore : ncore + ncas]
    hcore_ao = mf.get_hcore()
    hveff_ao = (0, 0)

    if ncore != 0:
        core_dmao = mo_core[0] @ mo_core[0].T.conj(), mo_core[1] @ mo_core[1].T.conj()
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj[0] + vj[1] - vk
        ecore0 = np.einsum("ij,ji->", core_dmao[0], hcore_ao + 0.5 * hveff_ao[0], optimize='optimal')
        ecore0 += np.einsum("ij,ji->", core_dmao[1], hcore_ao + 0.5 * hveff_ao[1], optimize='optimal')
    else:
        ecore0 = 0.0

    h1e_a = mo_cas[0].T.conj() @ (hcore_ao + hveff_ao[0]) @ mo_cas[0]
    h1e_b = mo_cas[1].T.conj() @ (hcore_ao + hveff_ao[1]) @ mo_cas[1]

    eri_ao = mol if mf._eri is None else mf._eri
    mo_a, mo_b = mo_cas
    g2e_aa = ao2mo.restore(1, ao2mo.full(eri_ao, mo_a), ncas)
    g2e_ab = ao2mo.restore(1, ao2mo.general(eri_ao, (mo_a, mo_a, mo_b, mo_b)), ncas)
    g2e_bb = ao2mo.restore(1, ao2mo.full(eri_ao, mo_b), ncas)

    xg2e_aa = g2e_aa.transpose(0, 2, 1, 3) - g2e_aa.transpose(0, 2, 3, 1)
    xg2e_bb = g2e_bb.transpose(0, 2, 1, 3) - g2e_bb.transpose(0, 2, 3, 1)
    xg2e_ab = g2e_ab.transpose(0, 2, 1, 3)

    # try to get 1e term without n.o. term
    #xg2e_aa = 0.0 * xg2e_aa
    #xg2e_ab = 0.0 * xg2e_ab
    #xg2e_bb = 0.0 * xg2e_bb

    #h1e_a = mo_cas[0].T.conj() @ hcore_ao @ mo_cas[0]
    #h1e_b = mo_cas[1].T.conj() @ hcore_ao @ mo_cas[1]
    # 

    return (h1e_a, h1e_b, xg2e_aa, xg2e_ab, xg2e_bb), (nocca, noccb), (nvira, nvirb)

# compute <ci|det>
def evaluate_ovlp(mf, ci_amps, det_mo_coeff):
    import numpy as np
    t_ord = len(ci_amps) - 1
    tensor_eqs = [cisd_eqs_ovlp, cisdt_eqs_ovlp, cisdtq_eqs_ovlp][t_ord - 2]
    metric = np.array(mf.get_ovlp())
    nocca, noccb = (mf.mol.nelectron + mf.mol.spin) // 2, (mf.mol.nelectron - mf.mol.spin) // 2
    cxs = mf.mo_coeff
    cws = det_mo_coeff
    cx_occs = np.array(cxs[0][:, :nocca]), np.array(cxs[1][:, :noccb])
    cw_occs = np.array(cws[0][:, :nocca]), np.array(cws[1][:, :noccb])
    rvlp, xmats, ymats = 1.0, [], []
    for ix, (cx_occ, cw_occ, cx) in enumerate(zip(cx_occs, cw_occs, cxs)):
        sxx, cx_occ, cw_occ = lowdin_pair(cx_occ, cw_occ, metric)
        inv_sxx, reduced_ovlp, zeros = reduced_overlap(sxx)
        mmat = cw_occ @ np.diag(inv_sxx) @ cx_occ.T
        mmat += cx_occ[:, zeros] @ cx_occ[:, zeros].T
        xmat_xx = cx.T @ metric @ mmat @ metric @ cx
        smat_xx = cx.T @ metric @ cx
        ymat_xx = smat_xx - xmat_xx
        rvlp *= reduced_ovlp
        xmats.append(xmat_xx)
        ymats.append(ymat_xx)
    
    nocca = (mf.mol.nelectron + mf.mol.spin) // 2
    noccb = (mf.mol.nelectron - mf.mol.spin) // 2
    nvira, nvirb = mf.mo_coeff[0].shape[1] - nocca, mf.mo_coeff[0].shape[1] - noccb
    n_occ, n_virt = (nocca, noccb), (nvira, nvirb)

    new_ci_amps = [list(x) for x in ci_amps]
    for ix, xcis in enumerate(ci_amps):
        for k, xci in enumerate(xcis):
            assert xci.shape == (n_occ[0], ) * (ix - k) + (n_occ[1], ) * k + (n_virt[0], ) * (ix - k) + (n_virt[1], ) * k
            new_ci_amps[ix][k] = np.zeros((n_occ[0] + n_virt[0], ) * (ix + ix))
            xx = xci.transpose(*range(ix, ix + ix), *range(ix))
            new_ci_amps[ix][k][(slice(n_occ[0], n_occ[0] + n_virt[0]), ) * (ix - k)
                + (slice(n_occ[1], n_occ[1] + n_virt[1]), ) * k
                + (slice(n_occ[0]), ) * (ix - k) + (slice(n_occ[0]), ) * k] = xx

    prx = lambda idx: (idx.count('I') + idx.count('E')) // 2

    result = 0
    for f, script, idx, nm in tensor_eqs:
        tensors = []
        for inm, iidx in zip(nm[:-1], idx[:-1]):
            if inm == 'T':
                tensors.append(new_ci_amps[len(iidx) // 2][prx(iidx)])
            elif inm == 'A':
                tensors.append({'ii': xmats[0].T, 'ee': ymats[0].T}[iidx])
            elif inm == 'B':
                tensors.append({'II': xmats[1].T, 'EE': ymats[1].T}[iidx])
        result += f * np.einsum(script, *tensors, optimize=True)
    return result * rvlp

# compute <ci|H|det>
def evaluate(mf, ci_amps, det_mo_coeff):
    import numpy as np
    t_ord = len(ci_amps) - 1
    print(f"T order: {t_ord}")
    tensor_eqs = [cisd_eqs, cisdt_eqs, cisdtq_eqs][t_ord - 2]
    metric = np.array(mf.get_ovlp())
    nocca, noccb = (mf.mol.nelectron + mf.mol.spin) // 2, (mf.mol.nelectron - mf.mol.spin) // 2
    cxs = mf.mo_coeff
    cws = det_mo_coeff
    cx_occs = np.array(cxs[0][:, :nocca]), np.array(cxs[1][:, :noccb])
    cw_occs = np.array(cws[0][:, :nocca]), np.array(cws[1][:, :noccb])
    rvlp, xmats, ymats = 1.0, [], []
    for ix, (cx_occ, cw_occ, cx) in enumerate(zip(cx_occs, cw_occs, cxs)):
        sxx, cx_occ, cw_occ = lowdin_pair(cx_occ, cw_occ, metric)
        inv_sxx, reduced_ovlp, zeros = reduced_overlap(sxx)
        mmat = cw_occ @ np.diag(inv_sxx) @ cx_occ.T
        mmat += cx_occ[:, zeros] @ cx_occ[:, zeros].T
        xmat_xx = cx.T @ metric @ mmat @ metric @ cx
        smat_xx = cx.T @ metric @ cx
        ymat_xx = smat_xx - xmat_xx
        rvlp *= reduced_ovlp
        xmats.append(xmat_xx)
        ymats.append(ymat_xx)
    ints, n_occ, n_virt = ut_ao2mo(mf)

    new_ci_amps = [list(x) for x in ci_amps]
    for ix, xcis in enumerate(ci_amps):
        for k, xci in enumerate(xcis):
            print(xci.shape)
            assert xci.shape == (n_occ[0], ) * (ix - k) + (n_occ[1], ) * k + (n_virt[0], ) * (ix - k) + (n_virt[1], ) * k
            new_ci_amps[ix][k] = np.zeros((n_occ[0] + n_virt[0], ) * (ix + ix))
            xx = xci.transpose(*range(ix, ix + ix), *range(ix))
            new_ci_amps[ix][k][(slice(n_occ[0], n_occ[0] + n_virt[0]), ) * (ix - k)
                + (slice(n_occ[1], n_occ[1] + n_virt[1]), ) * k
                + (slice(n_occ[0]), ) * (ix - k) + (slice(n_occ[1]), ) * k] = xx

    prx = lambda idx: (idx.count('I') + idx.count('E')) // 2
    iprx = lambda idx: (len(idx) // 2) * (len(idx) // 2 + 1) // 2 - 1 + prx(idx)

    result = 0
    for f, script, idx, nm in tensor_eqs:
        tensors = []
        for inm, iidx in zip(nm[:-1], idx[:-1]):
            if inm == 'T':
                tensors.append(new_ci_amps[len(iidx) // 2][prx(iidx)])
            elif inm == 'H':
                tensors.append(ints[iprx(iidx)])
            elif inm == 'A':
                tensors.append({'ii': xmats[0].T, 'ee': ymats[0].T}[iidx])
            elif inm == 'B':
                tensors.append({'II': xmats[1].T, 'EE': ymats[1].T}[iidx])
        result += f * np.einsum(script, *tensors, optimize=True)
    return result * rvlp