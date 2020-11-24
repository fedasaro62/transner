#/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
python usage.py --strings\
            "Robert King served in the Peace Corps and traveled extensively before completing his degree in English at the University of Michigan in 1992, the year he joined the company.  
            After completing a course entitled \"Selling in Europe,\" he was transferred to the London office in March 1993.
            Laura received a BA in psychology from the University of Washington. 
            She has also completed a course in business French.  
            She reads and writes French. 
            Anne has a BA degree in English from St. Lawrence College. 
            She is fluent in French and German.
            Michael is a graduate of Sussex University (MA, economics, 1983) and the University of California at Los Angeles (MBA, marketing, 1986). 
            He has also taken the courses \"Multi-Cultural Selling\" and \"Time Management for the Sales Professional.\" 
            He is fluent in Japanese and can read and write French, Portuguese, and Spanish.
            Steven Buchanan graduated from St. Andrews University, Scotland, with a BSC degree in 1976. 
            Upon joining the company as a sales representative in 1992, he spent 6 months in an orientation program at the Seattle office and then returned to his permanent post in London. 
            He was promoted to sales manager in March 1993. Mr. Buchanan has completed the courses \"Successful Telemarketing\" and \"International Sales Management.\" 
            He is fluent in French. Margaret holds a BA in English literature from Concordia College (1958) and an MA from the American Institute of Culinary Arts (1966).
            She was assigned to the London office temporarily from July through November 1992.
            The young Feynman was heavily influenced by his father, who encouraged him to ask questions to challenge orthodox thinking, 
            and who was always ready to teach Feynman something new. 
            From his mother, he gained the sense of humor that he had throughout his life. 
            As a child, he had a talent for engineering, maintained an experimental laboratory in his home, 
            and delighted in repairing radios. This radio repairing was probably the first job Feynman had had, 
            and during this time he showed early signs of an aptitude for his later career in theoretical physics, 
            when he would analyze the issues theoretically and arrive at the solutions[11]. 
            When he was in grade school, he created a home burglar alarm system while his parents were out for the day running errands.
            When Richard was five, his mother gave birth to a younger brother, Henry Phillips, who died at age four weeks.
            Four years later, Richard's sister Joan was born and the family moved to Far Rockaway, Queens. 
            Though separated by nine years, Joan and Richard were close, and they both shared a curiosity about the world. 
            Though their mother thought women lacked the capacity to understand such things, 
            Richard encouraged Joan's interest in astronomy, and Joan eventually became an astrophysicist."\
            --cuda\
