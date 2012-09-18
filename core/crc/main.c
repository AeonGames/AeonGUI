/**********************************************************************
 *
 * Filename:    main.c
 * 
 * Description: A simple test program for the CRC implementations.
 *
 * Notes:       To test a different CRC standard, modify crc.h.
 *
 * 
 * Copyright (c) 2000 by Michael Barr.  This software is placed into
 * the public domain and may be used for any purpose.  However, this
 * notice must not be changed or removed and no warranty is either
 * expressed or implied by its publication or distribution.
 **********************************************************************/

#include <stdio.h>
#include <string.h>

#include "crc.h"

extern crc crcTable[];

void
printTable(void)
{
    int x,y;
    printf("crc crcTable[256]  = {\n");
    for(y=0;y<64;++y)
    {
        for(x=0;x<4;++x)
        {
#ifdef CRC32
            printf("0x%08x,",crcTable[(y*4)+x]);
#else
            printf("0x%04x,",crcTable[(y*4)+x]);
#endif
        }
        printf("\n");
    }
    printf("}\n");
}


void
main(void)
{
	unsigned char  test[] = "123456789";


	/*
	 * Print the check value for the selected CRC algorithm.
	 */
	printf("The check value for the %s standard is 0x%X\n", CRC_NAME, CHECK_VALUE);
	
	/*
	 * Compute the CRC of the test message, slowly.
	 */
	printf("The crcSlow() of \"123456789\" is 0x%X\n", crcSlow(test, strlen(test)));
	
	/*
	 * Compute the CRC of the test message, more efficiently.
	 */
	crcInit();
	printf("The crcFast() of \"123456789\" is 0x%X\n", crcFast(test, strlen(test)));
    printTable();
}   /* main() */