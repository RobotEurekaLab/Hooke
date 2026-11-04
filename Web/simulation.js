// Simulation showcase page — static gallery, no framework needed.
// Video paths are relative to Web/, resolving into ../docs/assets/ where
// the actual preview clips live (checked into the repo, so they load the
// same whether served from GitHub Pages or a local http.server).

const REAL_LAB = [
  { school: 'King Fahd University of Petroleum & Minerals', qs: '101', file: 'king-fahd-university-of-petroleum-minerals', room: false,
    detail: 'A field-emission SEM with a swinging EDX detector arm in the Chemistry Department Microscopy Laboratory, Building 4 Room 157, King Fahd University of Petroleum and Minerals.' },
  { school: 'KIT, Karlsruhe Institute of Technology', qs: '102', file: 'kit-karlsruhe-institute-of-technology', room: false,
    detail: 'A transmission electron microscope with a tilting goniometer specimen stage in the Laboratory for Electron Microscopy, Building 30.22 Room 228, Karlsruhe Institute of Technology.' },
  { school: 'Uppsala University', qs: '103', file: 'uppsala-university', room: false,
    detail: 'A Leica DIVE-style multiphoton microscope with a rotating filter turret in the BioVis imaging core facility, Rudbeck Laboratory, Uppsala University.' },
  { school: 'University of St Andrews', qs: '104', file: 'university-of-st-andrews', room: true,
    detail: 'A Thermo Orbitrap Exploris-style mass spectrometer with a hinged sample-plate loading door in the School of Chemistry analytical facilities, Purdie Building, University of St Andrews.' },
  { school: 'Utrecht University', qs: '105', file: 'utrecht-university', room: false,
    detail: 'A dual-beam FIB-SEM with a hinged specimen chamber door in the Cell Microscopy Core, Room H02.313, Utrecht University.' },
  { school: 'The University of Sheffield', qs: '105', file: 'the-university-of-sheffield', room: false,
    detail: 'A 10x Genomics Chromium Controller with a hinged chip-loading lid in the Multiomics Facility, Sheffield Institute for Translational Neuroscience, University of Sheffield.' },
  { school: 'Tohoku University', qs: '107', file: 'tohoku-university', room: false,
    detail: 'A 3.0T MRI scanner bore with a sliding patient table in the ToMMo biobank/genome-medicine facility, Seiryo campus, Tohoku University.' },
  { school: 'Boston University', qs: '108', file: 'boston-university', room: true,
    detail: 'A Nikon-style deconvolution wide-field epifluorescence microscope with a rotating filter turret in the Cellular Imaging Core, Evans Biomedical Research Center Basement B15, Boston University.' },
  { school: 'University of Nottingham', qs: '108', file: 'university-of-nottingham', room: false,
    detail: 'A genomics array scanner with a hinged sample-tray lid in the Post-Genomic Technologies Facility, A Floor West Block, Queen\'s Medical Centre, University of Nottingham.' },
  { school: 'Technical University of Denmark', qs: '109', file: 'technical-university-of-denmark', room: true,
    detail: 'A dual-beam FIB-SEM with a hinged specimen chamber door in the DTU Nanolab cleanroom, Building 347, Technical University of Denmark.' },
  { school: 'Politecnico di Milano', qs: '111', file: 'politecnico-di-milano', room: true,
    detail: 'A continuous-flow chemical reactor with a swinging valve lever in BiocatLab, Politecnico di Milano.' },
  { school: 'University of Waterloo', qs: '115', file: 'university-of-waterloo', room: false,
    detail: 'A gel-imaging system with a hinged UV-transilluminator lid in the Molecular Biology Core Facility, Room B1-371, Biology 1 building, University of Waterloo.' },
  { school: 'University of Wisconsin-Madison', qs: '116', file: 'university-of-wisconsin-madison', room: true,
    detail: 'An Illumina NovaSeq X Plus-style sequencer with a hinged loading door in the UW Biotech Center DNA Sequencing Facility, Room 1250, 425 Henry Mall, University of Wisconsin-Madison.' },
  { school: 'University of Helsinki', qs: '117', file: 'university-of-helsinki', room: false,
    detail: 'A benchtop DNA sequencer with a hinged flow-cell loading lid in the DNA Sequencing and Genomics Laboratory.' },
  { school: 'Indian Institute of Technology Bombay (IITB)', qs: '118', file: 'indian-institute-of-technology-bombay-iitb', room: false,
    detail: 'A 500 MHz NMR spectrometer with a hinged sample-insertion port in the IOE facility, Room 212, Department of Chemistry, IIT Bombay.' },
  { school: 'University of Oslo', qs: '119', file: 'university-of-oslo', room: false,
    detail: 'A live-cell confocal imaging microscope with a rotating filter turret in the Advanced Light Microscopy core facility, Rikshospitalet, University of Oslo.' },
  { school: 'Western University', qs: '120', file: 'western-university', room: true,
    detail: 'A real-time PCR system in the Molecular Genetics Unit, Room 357, Western Science Centre, Western University.' },
  { school: 'Queen Mary University of London', qs: '120', file: 'queen-mary-university-of-london', room: true,
    detail: 'A multi-capillary DNA sequencer with a hinged loading door in the Barts and the London Genome Centre, Queen Mary University of London.' },
  { school: 'RMIT University', qs: '123', file: 'rmit-university', room: false,
    detail: 'A JEOL/FEI-style dual-beam FIB-SEM with a hinged specimen chamber door in the RMIT Microscopy and Microanalysis Facility, Building 14, RMIT University.' },
  { school: 'University of Southern California', qs: '125', file: 'university-of-southern-california', room: true,
    detail: 'A benchtop genomics instrument with a hinged loading lid in the Molecular Genomics Core.' },
  { school: 'University College Dublin', qs: '126', file: 'university-college-dublin', room: true,
    detail: 'A Beckman Coulter-style flow cytometer with a hinged sample access lid in the Conway Genomics and Imaging Core, University College Dublin.' },
  { school: 'Stockholm University', qs: '128', file: 'stockholm-university', room: false,
    detail: 'A spark plasma sintering press with a moving upper ram in the Spark Plasma Sintering Facility, Arrhenius Laboratory, Stockholm University.' },
  { school: 'University of Basel', qs: '131', file: 'university-of-basel', room: false,
    detail: 'A Leica LMD7-style laser microdissection microscope with a rotating objective/laser-path turret in the Imaging Core Facility, Biozentrum, University of Basel.' },
  { school: 'Eindhoven University of Technology', qs: '136', file: 'eindhoven-university-of-technology', room: true,
    detail: 'A femtosecond-laser micromachining setup with a swinging optics arm in the TU/e Microfab Lab, Building 15 Gemini-Noord, Eindhoven University of Technology.' },
  { school: 'Universiti Kebangsaan Malaysia (UKM)', qs: '138', file: 'universiti-kebangsaan-malaysia-ukm', room: true,
    detail: 'A gradient PCR thermocycler with a hinged heating-block lid in the Genomics Lab, INBIOSIS, Universiti Kebangsaan Malaysia.' },
  { school: 'Universidad de Chile', qs: '139', file: 'universidad-de-chile', room: true,
    detail: 'A PCR thermocycler with a hinged heating-block lid in the Laboratorio de Biologia Molecular e Ingenieria Genetica, Universidad de Chile.' },
  { school: 'Lancaster University', qs: '141', file: 'lancaster-university', room: true,
    detail: 'An inverted confocal microscope in the Advanced Light Microscopy Facility, Division of Biomedical and Life Sciences, Lancaster University.' },
  { school: 'Rice University', qs: '141', file: 'rice-university', room: true,
    detail: 'A photolithography mask aligner chamber in the Rice Nanofabrication Facility, Space Science and Technology Building, Rice University.' },
  { school: 'Leiden University', qs: '141', file: 'leiden-university', room: false,
    detail: 'A Titan Krios-style cryo-electron microscope with a cryo-dewar and hinged specimen airlock in the NeCEN facility, Gorlaeus Laboratory, Leiden University.' },
  { school: 'Aarhus University', qs: '144', file: 'aarhus-university', room: false,
    detail: 'A spinning-disk confocal microscope with a rotating objective turret in the Bioimaging Core Facility, Skou Building 1116 Room 256, Aarhus University.' },
  { school: 'Universiti Sains Malaysia (USM)', qs: '146', file: 'universiti-sains-malaysia-usm', room: false,
    detail: 'A pulsed-field gel electrophoresis (PFGE) system with a hinged buffer-chamber lid in the INFORMM equipment facility, Universiti Sains Malaysia.' },
  { school: 'Technische Universität Berlin (TU Berlin)', qs: '147', file: 'technische-universit-t-berlin-tu-berlin', room: false,
    detail: 'A benchtop flow cytometer with a hinged sample-tube access lid in the Chair of Environmental Microbiomics laboratory, Room BH 6-1, TU Berlin.' },
  { school: 'Nagoya University', qs: '152', file: 'nagoya-university', room: true,
    detail: 'A PALM Combi laser microdissection and optical-tweezers cell-manipulation microscope with a swinging laser delivery arm in the ITbM Live Imaging Center, Nagoya University.' },
  { school: 'Michigan State University', qs: '152', file: 'michigan-state-university', room: false,
    detail: 'An ABI QuantStudio 7 Flex real-time PCR system with a hinged sample-plate loading door in the RTSF Genomics Core, Michigan State University.' },
  { school: 'University of Geneva', qs: '155', file: 'university-of-geneva', room: false,
    detail: 'An Olympus VS120-style slide scanner with a sliding slide tray in Room C06.1533.a, Bioimaging Core Facility, CMU Building C, University of Geneva.' },
  { school: 'University of North Carolina, Chapel Hill', qs: '155', file: 'university-of-north-carolina-chapel-hill', room: false,
    detail: 'A light-sheet fluorescence microscope with a translating sample mount in the Biology Microscopy Core, Genome Sciences Building Room 1152, University of North Carolina at Chapel Hill.' },
  { school: 'Erasmus University Rotterdam', qs: '158', file: 'erasmus-university-rotterdam', room: false,
    detail: 'A benchtop genomics sequencer with a hinged loading lid in the Erasmus Center for Biomics, Department of Molecular Genetics, Erasmus MC.' },
  { school: 'University of Groningen', qs: '159', file: 'university-of-groningen', room: true,
    detail: 'A SORP BD FACSAria-style flow cytometry cell sorter with a swinging sample probe arm in the GBB Dedicated Research Facilities, Linnaeusborg, University of Groningen.' },
  { school: 'University of Bern', qs: '161', file: 'university-of-bern', room: false,
    detail: 'A Zeiss LSM710-style confocal laser scanning microscope with a rotating filter turret in the Microscopy Imaging Center.' },
  { school: 'Kyushu University', qs: '167', file: 'kyushu-university', room: true,
    detail: 'An ultrafast laser spectroscopy bench with a swinging optics arm in the Laboratory of Spectrochemistry, Room B1009, Ito Campus, Kyushu University.' },
  { school: 'The University of Exeter', qs: '169', file: 'the-university-of-exeter', room: true,
    detail: 'An electron microscope with a swinging EDX detector arm in the Bioimaging Centre, Geoffrey Pope Building, University of Exeter.' },
  { school: 'University of Cape Town', qs: '171', file: 'university-of-cape-town', room: false,
    detail: 'A Class II biosafety cabinet with a sliding sash in the Institute of Infectious Disease and Molecular Medicine.' },
  { school: 'Curtin University', qs: '174', file: 'curtin-university', room: true,
    detail: 'An electron microscope with a specimen load-lock chamber in the John de Laeter Centre, Curtin University.' },
  { school: 'McMaster University', qs: '176', file: 'mcmaster-university', room: false,
    detail: 'A PacBio-style long-read DNA sequencer with a hinged loading door in the McMaster Genomics Facility, Room 3N4, Health Sciences Centre, McMaster University.' },
  { school: 'Washington University in St. Louis', qs: '176', file: 'washington-university-in-st-louis', room: true,
    detail: 'A wide-field microscope for histology and fluorescence imaging with a rotating filter turret in the Molecular Microbiology Imaging Facility, Room 10302, McDonnell Pediatric Research Building, Washington University in St. Louis.' },
  { school: 'University of California, Santa Barbara (UCSB)', qs: '178', file: 'university-of-california-santa-barbara-ucsb', room: false,
    detail: 'A Nanolive 3D Cell Explorer label-free live-cell tomography microscope with a rotating filter turret in the NRI-MCDB Microscopy Facility, Bio2 Building Room 5173B, UC Santa Barbara.' },
  { school: 'The University of Newcastle, Australia (UON)', qs: '179', file: 'the-university-of-newcastle-australia-uon', room: false,
    detail: 'A Thermo Q Exactive Orbitrap mass spectrometer with a hinged sample-plate loading door in the Central Analytical Facilities.' },
  { school: 'Universidad de los Andes', qs: '179', file: 'universidad-de-los-andes', room: false,
    detail: 'A Tescan Lyra 3 dual-beam FIB-SEM with a hinged specimen chamber door in the Centro de Microscopia.' },
  { school: 'Waseda University', qs: '181', file: 'waseda-university', room: false,
    detail: 'A JEOL JEM-2100F field-emission STEM with a tilting goniometer stage in the Analytical Instrument Laboratory, Building 42-1 Room 212, Waseda University.' },
  { school: 'Hamad Bin Khalifa University', qs: '183', file: 'hamad-bin-khalifa-university', room: false,
    detail: 'An Amnis ImageStream MKII imaging flow cytometer with a hinged sample access lid in the QBRI Imaging and Flow Cytometry Core, Hamad Bin Khalifa University.' },
  { school: 'University of York', qs: '184', file: 'university-of-york', room: true,
    detail: 'A Carl Zeiss ELYRA 7 super-resolution SIM/PALM/STORM microscope with a rotating filter turret in the Technology Facility Imaging and Cytometry Laboratory, University of York.' },
  { school: 'Keio University', qs: '188', file: 'keio-university', room: false,
    detail: 'A Zeiss MultiSEM 505 multi-beam scanning electron microscope with a swinging EDX detector arm in the Electron Microscope Center, Preventive Medicine Building, Keio University.' },
  { school: 'University of Ottawa', qs: '189', file: 'university-of-ottawa', room: false,
    detail: 'A Leica TCS SP5-STED super-resolution confocal microscope with a rotating filter turret in the CBIA Core Facility, 451 Smyth Road, University of Ottawa.' },
  { school: 'Technische Universität Wien', qs: '190', file: 'technische-universit-t-wien', room: false,
    detail: 'A JEOL NeoARM 200 aberration-corrected TEM with a tilting goniometer stage in USTEM, Room 057-02, Freihaus building, TU Wien.' },
  { school: 'Universität Hamburg', qs: '191', file: 'universit-t-hamburg', room: false,
    detail: 'A Leica/Nikon/Zeiss-style confocal microscope with a rotating filter turret in the Technology Platform Light Microscopy, Universitat Hamburg.' },
  { school: 'Queen\'s University at Kingston', qs: '193', file: 'queen-s-university-at-kingston', room: false,
    detail: 'A Bruker AutoFlex-style MALDI-TOF mass spectrometer with a hinged sample-plate loading door in the Department of Chemistry shared analytical facilities, Chernoff Hall, Queen\'s University at Kingston.' },
  { school: 'University of Gothenburg', qs: '194', file: 'university-of-gothenburg', room: false,
    detail: 'A MALDI imaging mass spectrometer with a hinged sample-plate loading door in the Centre for Cellular Imaging, University of Gothenburg.' },
  { school: 'Wuhan University', qs: '194', file: 'wuhan-university', room: false,
    detail: 'A microplate reader with a hinged sample-tray lid on the biochemical and molecular detection platform, Shared Instrument Platform, College of Life Sciences, Wuhan University.' },
  { school: 'Emory University', qs: '196', file: 'emory-university', room: false,
    detail: 'A benchtop DNA sequencer with a slide-out flow-cell drawer in the Emory Integrated Genomics Core, Woodruff Memorial Research Building, Emory University.' },
  { school: 'Deakin University', qs: '197', file: 'deakin-university', room: false,
    detail: 'A JEOL IT 300-style scanning electron microscope with a swinging EDX detector arm in the Materials Science Labs, Deakin University.' },
  { school: 'University of Calgary', qs: '198', file: 'university-of-calgary', room: false,
    detail: 'A Bruker NanoWizard IV atomic force microscope with a vertical scan head in the Microscopy and Imaging Facility.' },
  { school: 'Universidad Autónoma de Madrid', qs: '198', file: 'universidad-aut-noma-de-madrid', room: false,
    detail: 'A Leica Stellaris 8 confocal microscope with FLIM/STED modules and a rotating filter turret in the SMOA Advanced Optical Microscopy Facility, Lab 310, CBM, Universidad Autonoma de Madrid.' },
  { school: 'Arizona State University', qs: '200', file: 'arizona-state-university', room: false,
    detail: 'A Beckman Biomek-style automated liquid-handling gantry robot in the Desert Southwest Genomics Center, Arizona State University.' },
  { school: 'King Saud University', qs: '200', file: 'king-saud-university', room: false,
    detail: 'A Gammacell 220 sample irradiator with a sliding sample drawer in the Central Laboratory, College of Science, King Saud University.' },
  { school: 'Khalifa University of Science and Technology', qs: '202', file: 'khalifa-university-of-science-and-technology', room: false,
    detail: 'A transmission electron microscope with a hinged specimen airlock port in the Electron Microscopy Facility, Building L Room 1020, Khalifa University.' },
  { school: 'University of Minnesota Twin Cities', qs: '203', file: 'university-of-minnesota-twin-cities', room: false,
    detail: 'A LUMICKS C-Trap optical-tweezers instrument with a sliding microfluidic chip stage in the University Imaging Centers, Jackson Hall, University of Minnesota Twin Cities.' },
  { school: 'Université catholique de Louvain (UCLouvain)', qs: '203', file: 'universit-catholique-de-louvain-uclouvain', room: false,
    detail: 'A 600 MHz FT-NMR spectrometer with a hinged sample-insertion port in the NMR Platform, IMCN, Universite Catholique de Louvain.' },
  { school: 'The Ohio State University', qs: '208', file: 'the-ohio-state-university', room: false,
    detail: 'A Miltenyi UltraMicroscope Blaze light-sheet microscope in Room 245A, Campus Microscopy and Imaging Facility.' },
  { school: 'Tel Aviv University', qs: '209', file: 'tel-aviv-university', room: false,
    detail: 'A Zeiss LSM 510-META confocal laser scanning microscope with a rotating filter turret in the Rosalie and Harold Rae Brown Cancer Research Core Facility, Tel Aviv University.' },
  { school: 'National Tsing Hua University', qs: '210', file: 'national-tsing-hua-university', room: false,
    detail: 'A ZEISS LSM 800 confocal microscope with Airyscan and a rotating filter turret in the Confocal Microscope System.' },
  { school: 'Indian Institute of Science', qs: '211', file: 'indian-institute-of-science', room: false,
    detail: 'An atom probe tomography (APT) system with a tilting specimen stage in the Advanced Facility for Microscopy and Microanalysis (AFMM), Indian Institute of Science.' },
  { school: 'Albert-Ludwigs-Universitaet Freiburg', qs: '212', file: 'albert-ludwigs-universitaet-freiburg', room: false,
    detail: 'A Leica TCS SP8 STED 3x super-resolution microscope with a rotating filter turret in the Life Imaging Center.' },
  { school: 'Queensland University of Technology (QUT)', qs: '213', file: 'queensland-university-of-technology-qut', room: true,
    detail: 'A Zeiss Helios Helium Ion Microscope with a hinged specimen chamber door in the Central Analytical Research Facility.' },
  { school: 'University of Otago', qs: '214', file: 'university-of-otago', room: false,
    detail: 'A JEOL JSM-6700F field-emission SEM with a cryo-preparation stage and a swinging EDX detector arm in the OMNI electron microscopy unit, Room B10, Lindo Ferguson Building, University of Otago.' },
  { school: 'National Cheng Kung University (NCKU)', qs: '215', file: 'national-cheng-kung-university-ncku', room: true,
    detail: 'A JEOL JEM-1400 transmission electron microscope with a tilting goniometer stage in the TEM service, Room 82-B139, Medical Building, National Cheng Kung University.' },
  { school: 'University of Florida', qs: '215', file: 'university-of-florida', room: true,
    detail: 'A BD Symphony S6-style flow cytometer/sorter with a hinged sample access lid in the Cytometry & Optical Microscopy Core, Cancer & Genetics Research Complex, University of Florida.' },
  { school: 'University of Maryland, College Park', qs: '218', file: 'university-of-maryland-college-park', room: false,
    detail: 'A JPK NanoWizard 4a atomic force microscope with a vertical scan head in the CMNS Imaging Incubator, Physical Sciences Complex, University of Maryland, College Park.' },
  { school: 'National Yang Ming Chiao Tung University', qs: '219', file: 'national-yang-ming-chiao-tung-university', room: false,
    detail: 'A Carl Zeiss LSM900 confocal microscope with Airyscan2 and a rotating filter turret in the Imaging Core Facility, Room 639, Library Building, Yang Ming Campus, National Yang Ming Chiao Tung University.' },
  { school: 'Eberhard Karls Universität Tübingen', qs: '222', file: 'eberhard-karls-universit-t-t-bingen', room: false,
    detail: 'An electron microscope with a swinging EDX detector arm in the Tubingen Structural Microscopy.' },
  { school: 'Indian Institute of Technology Kharagpur (IIT-KGP)', qs: '222', file: 'indian-institute-of-technology-kharagpur-iit-kgp', room: false,
    detail: 'A ZEISS/JEOL-style field-emission SEM with a hinged specimen chamber door in the Central Research Facility.' },
  { school: 'Loughborough University', qs: '224', file: 'loughborough-university', room: false,
    detail: 'A benchtop NMR spectrometer with a hinged sample port in the WPL.2.09 Chemistry Synthesis Laboratory, STEMLab building, Loughborough University.' },
  { school: 'Friedrich-Alexander-Universität Erlangen-Nürnberg', qs: '224', file: 'friedrich-alexander-universit-t-erlangen-n-rnberg', room: false,
    detail: 'A dual-beam electron microscope with a hinged specimen chamber door in CENEM, the Center for Nanoanalysis and Electron Microscopy, FAU Erlangen-Nurnberg.' },
  { school: 'Rheinische Friedrich-Wilhelms-Universität Bonn', qs: '227', file: 'rheinische-friedrich-wilhelms-universit-t-bonn', room: false,
    detail: 'An Abberior easy3D STED super-resolution microscope with a rotating filter turret in the LIMES Technical Platforms, University of Bonn.' },
  { school: 'Indian Institute of Technology Madras (IITM)', qs: '227', file: 'indian-institute-of-technology-madras-iitm', room: false,
    detail: 'A benchtop fermenter with a hinged vessel lid in the IITM Bioincubator, IIT Madras Research Park.' },
  { school: 'Chulalongkorn University', qs: '229', file: 'chulalongkorn-university', room: false,
    detail: 'A GAMRY-style benchtop potentiostat with a hinged electrode access lid in the Energetic Materials Research Laboratory, Chulalongkorn University.' },
  { school: 'Universite libre de Bruxelles', qs: '230', file: 'universite-libre-de-bruxelles', room: false,
    detail: 'A Zeiss LSM780-style confocal/multiphoton microscope with a rotating filter turret in the Light Microscopy Facility.' },
  { school: 'Universidade Estadual de Campinas (Unicamp)', qs: '232', file: 'universidade-estadual-de-campinas-unicamp', room: false,
    detail: 'An atomic force microscope with a vertical scan head in the LIMicro-IQ microscopy core facility, Room D106, Institute of Chemistry, Universidade Estadual de Campinas.' },
  { school: 'University of Twente', qs: '233', file: 'university-of-twente', room: false,
    detail: 'A thin-film deposition chamber with a hinged viewport hatch in the MESA+ Institute NanoLab cleanroom, University of Twente.' },
  { school: 'University of Rochester', qs: '236', file: 'university-of-rochester', room: false,
    detail: 'A Covaris-style focused-ultrasonicator with a hinged water-bath lid in the Rochester Genomics Center, James P. Wilmot Cancer Institute, University of Rochester.' },
  { school: 'Università di Padova', qs: '236', file: 'universit-di-padova', room: false,
    detail: 'A FEI Tecnai G2-style transmission electron microscope with a tilting goniometer specimen stage in the DiBio Imaging Facility, University of Padova.' },
  { school: 'University of Aberdeen', qs: '236', file: 'university-of-aberdeen', room: false,
    detail: 'A ZEISS Celldiscoverer 7 high-content imaging platform with a robotic plate-handling arm in the Microscopy and Histology Core Facility, Institute of Medical Sciences, University of Aberdeen.' },
  { school: 'Gadjah Mada University', qs: '239', file: 'gadjah-mada-university', room: false,
    detail: 'An Oxford Nanopore PromethION 24-style sequencer with a hinged loading lid in the Integrated Genome Factory.' },
  { school: 'Massey University', qs: '239', file: 'massey-university', room: true,
    detail: 'An ABI 3500xl-style capillary genetic analyzer with a hinged loading door in the Massey Genome Service, Room ScD3.15A, Turitea Campus, Massey University.' },
  { school: 'Technical University of Darmstadt', qs: '241', file: 'technical-university-of-darmstadt', room: true,
    detail: 'A JEOL ARM 200F aberration-corrected STEM with a tilting goniometer stage in the Advanced Electron Microscopy Division, Building L2|01 Room 52, TU Darmstadt.' },
  { school: 'Politecnico di Torino', qs: '241', file: 'politecnico-di-torino', room: false,
    detail: 'A Vibracell VC-505 ultrasonic processor with a hinged lid in the Chemical Synthesis Laboratory, Technological Centre in Alessandria, Politecnico di Torino.' },
  { school: 'Dartmouth College', qs: '243', file: 'dartmouth-college', room: false,
    detail: 'An Andor Dragonfly-style spinning-disk confocal microscope with a rotating filter turret in the Life Sciences Light Microscopy Facility, Class of 1978 Life Sciences Center, Dartmouth College.' },
  { school: 'Victoria University of Wellington', qs: '244', file: 'victoria-university-of-wellington', room: true,
    detail: 'A flow-through seawater wet-lab bench with a swinging tap lever in the Victoria University Coastal Ecology Lab.' },
  { school: 'University of Sussex', qs: '246', file: 'university-of-sussex', room: false,
    detail: 'An Olympus ScanR automated high-content screening platform with a robotic plate-handling arm in the Wolfson Centre for Biological Imaging, University of Sussex.' },
  { school: 'Vanderbilt University', qs: '248', file: 'vanderbilt-university', room: false,
    detail: 'An environmental scanning electron microscope with a swinging EDX detector arm in the Cell Imaging Shared Resource, Vanderbilt University.' },
  { school: 'American University of Beirut (AUB)', qs: '250', file: 'american-university-of-beirut-aub', room: false,
    detail: 'A next-generation sequencing platform with a hinged loading lid in the Genomic Profiling Program, Aida and Halim Daniel Academic and Clinical Center, American University of Beirut.' },
];

const SPACE = [
  { label: 'Lunar', slug: 'lunar' },
  { label: 'Martian', slug: 'martian' },
  { label: 'Orbital', slug: 'orbital' },
];

const MICRO = [
  { file: 'microscopy-cell-manipulation', title: 'Grasp and transfer',
    detail: 'Independently driven fine forceps lift and relocate a 36 µm cell with <3 µm placement error.' },
  { file: 'microscopy-cell-injection', title: 'Volume-controlled injection',
    detail: 'Membrane puncture and 0.5 pL delivery into a 36 × 28 × 8 µm adherent cell.' },
  { file: 'microscopy-cell-pushing', title: 'Cell pushing',
    detail: 'A 4 µm blunt probe holds ≥50 ms contact within 3 µm of the target, ≤4.5 µm compression.' },
];

function el(tag, className, children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const child of children || []) node.append(child);
  return node;
}

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// With 100 real-lab clips on one page, playing all of them at once would mean
// 100 simultaneous decoders -- fine on a fast machine, not fine on most.
// Instead, only the cards actually scrolled into view are allowed to play;
// everything else holds on its first frame (or is paused) until it scrolls
// in. A single shared IntersectionObserver drives every video on the page.
const playbackObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      const video = entry.target;
      if (reduceMotion) return;
      if (entry.isIntersecting) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  },
  { rootMargin: '200px 0px', threshold: 0.01 }
);

function lazyVideo(src) {
  const video = document.createElement('video');
  video.src = src;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = 'metadata';
  video.addEventListener('click', () => (video.paused ? video.play() : video.pause()));
  playbackObserver.observe(video);
  return video;
}

let realLabCards = [];

function buildRealLabGrid() {
  const grid = document.getElementById('reallab-grid');
  if (!grid) return;
  realLabCards = REAL_LAB.map((lab) => {
    const card = el('a', 'video-card', []);
    card.href = `https://github.com/RobotEurekaLab/Hooke/blob/main/real_lab/labs/${lab.file}.md`;
    card.target = '_blank';
    card.rel = 'noopener';
    const frame = el('div', 'video-frame', [lazyVideo(`../docs/assets/real-lab-${lab.file}-mujoco.mp4`)]);
    const qs = el('span', 'qs-badge', []);
    qs.textContent = `QS ${lab.qs}`;
    frame.append(qs);
    if (lab.room) {
      const roomBadge = el('span', 'room-badge', []);
      roomBadge.textContent = 'Room rebuilt';
      frame.append(roomBadge);
    }
    const body = el('div', 'video-card-body', []);
    const h3 = el('h3', null, []); h3.textContent = lab.school;
    const p = el('p', null, []); p.textContent = lab.detail;
    body.append(h3, p);
    card.append(frame, body);
    grid.append(card);
    return { el: card, school: lab.school.toLowerCase(), room: lab.room };
  });
  applyRealLabFilter();
}

function applyRealLabFilter() {
  const searchInput = document.getElementById('reallab-search');
  const roomOnly = document.getElementById('reallab-room-only');
  const countEl = document.getElementById('reallab-count');
  const emptyEl = document.getElementById('reallab-empty');
  if (!searchInput || !roomOnly) return;
  const query = searchInput.value.trim().toLowerCase();
  const onlyRooms = roomOnly.checked;
  let shown = 0;
  for (const card of realLabCards) {
    const matches = (!query || card.school.includes(query)) && (!onlyRooms || card.room);
    card.el.hidden = !matches;
    if (matches) shown += 1;
  }
  if (countEl) countEl.textContent = `${shown} of ${realLabCards.length} universities`;
  if (emptyEl) emptyEl.hidden = shown !== 0;
}

function buildSpaceGrid() {
  const grid = document.getElementById('space-grid');
  if (!grid) return;
  for (const world of SPACE) {
    const row = el('div', 'pair-row', []);
    const label = el('div', 'pair-label', []);
    label.textContent = world.label;
    row.append(label);
    for (const backend of ['mujoco', 'isaac']) {
      const col = el('div', 'pair-col', []);
      const video = lazyVideo(`../docs/assets/space-experiment-${world.slug}-sample_transfer-${backend}.mp4`);
      const tag = el('span', 'backend-tag', []);
      tag.textContent = backend === 'mujoco' ? 'MuJoCo' : 'Isaac Sim';
      col.append(video, tag);
      row.append(col);
    }
    grid.append(row);
  }
}

function buildMicroGrid() {
  const grid = document.getElementById('micro-grid');
  if (!grid) return;
  for (const shot of MICRO) {
    const card = el('div', 'shot-card', []);
    const img = document.createElement('img');
    img.src = `../docs/assets/${shot.file}.png`;
    img.loading = 'lazy';
    img.alt = shot.title;
    const body = el('div', 'shot-card-body', []);
    const h3 = el('h3', null, []); h3.textContent = shot.title;
    const p = el('p', null, []); p.textContent = shot.detail;
    body.append(h3, p);
    card.append(img, body);
    grid.append(card);
  }
}

buildRealLabGrid();
buildSpaceGrid();
buildMicroGrid();

document.getElementById('reallab-search')?.addEventListener('input', applyRealLabFilter);
document.getElementById('reallab-room-only')?.addEventListener('change', applyRealLabFilter);

// Fade each section in as it enters view, same restrained reveal used on
// the landing page's screen 2.
const revealTargets = document.querySelectorAll('.sim-section, .sim-hero, .sim-closing');
revealTargets.forEach((sectionEl) => {
  sectionEl.style.opacity = '0';
  sectionEl.style.transform = 'translateY(22px)';
  sectionEl.style.transition = 'opacity .7s ease, transform .7s ease';
});
const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      io.unobserve(entry.target);
    });
  },
  // A small absolute threshold rather than a fraction of the target's own
  // height: the real-lab section alone is now ~100 cards tall, many times
  // the viewport height, so a ratio-based threshold (e.g. 0.12) can never
  // be reached at any scroll position and the section would stay invisible
  // forever. `threshold: 0` fires as soon as even one pixel is visible.
  { threshold: 0 }
);
revealTargets.forEach((sectionEl) => io.observe(sectionEl));
