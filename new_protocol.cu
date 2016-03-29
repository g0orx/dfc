/**
* @file new_protocol.cu
* @brief New Ethernet protocol
* @author John Melton, G0ORX/N6LYT
*/


/* Copyright (C)
* 2015 - John Melton, G0ORX/N6LYT
*
* Based on code by Steven Passe AD0ES and Vasiliy Gokoyev K3IT
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
*
*/

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/if_ether.h>
#include <netpacket/packet.h>
#include <net/if_packet.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "common.cuh"
#include "new_protocol.cuh"

#define STATE_IDLE 0
#define STATE_RUNNING 1
static int state;

#define BASE_PORT 1024
static int new_socket;
static unsigned char mac_address[6];

static long base_sequence=0L;

#define MAX_BUFFER_LEN 1444
#define DISCOVERY_BUFFER_LEN 60

void *new_read_thread(void *arg) {

    struct sockaddr_in read_addr;
    uint8_t read_buffer[MAX_BUFFER_LEN];
    socklen_t read_length;
    struct ifreq ifr;
    int rc;
    int on=1;

    new_socket = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (new_socket < 0) {
        perror("new_read_thread: create socket failed for new_socket\n");
        exit(EXIT_FAILURE);
    }

    rc = setsockopt(new_socket, SOL_SOCKET, SO_REUSEADDR, (const void*)&on, sizeof(on));
    if (rc != 0) {
        fprintf(stderr, "new_read_thread: cannot set SO_REUSEADDR: rc=%d\n", rc);
        exit(EXIT_FAILURE);
    }

    read_addr.sin_family = AF_INET;
    read_addr.sin_port = htons(BASE_PORT);
    read_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(new_socket, (struct sockaddr*) &read_addr, sizeof(read_addr)) < 0) {
        perror("new_read_thread: bind socket failed for new_socket\n");
        exit(EXIT_FAILURE);
    }

    rc = setsockopt(new_socket, SOL_SOCKET, SO_BROADCAST, (const void*)&on, sizeof(on));
    if (rc != 0) {
        fprintf(stderr, "new_read_thread: cannot set SO_BROADCAST: rc=%d\n", rc);
        exit(EXIT_FAILURE);
    }

    ifr.ifr_addr.sa_family = AF_INET;
    strncpy(ifr.ifr_name, interface, IFNAMSIZ-1);
    ioctl(new_socket, SIOCGIFADDR, &ifr);

    unsigned char* u = (unsigned char*)&ifr.ifr_addr.sa_data;
    for (int k = 0; k < 6; k++) {
        mac_address[k] = u[k];
    }

    while(1) {
        if ((rc=recvfrom(new_socket, read_buffer, sizeof(read_buffer), 0,
                      (struct sockaddr*)&read_addr, &read_length)) < 0) {
            fprintf(stderr, "new_read_thread: error recvfrom %d", rc);
            exit(EXIT_FAILURE);
        }

        short port=htons(read_addr.sin_port);
        unsigned long sequence;
        uint8_t command;

        if(port==BASE_PORT) {
            sequence=(read_buffer[0]<<24)+(read_buffer[1]<<16)+(read_buffer[2]<<8)+read_buffer[3];
            command=read_buffer[4];
            switch(command) {
                case 0:  // general packet
                    break;
                case 2:  // discovery
                    // check length==DISCOVERY_BUFFER_LEN?
                    // send reply
                    uint8_t discovery_reply[DISCOVERY_BUFFER_LEN];
                    memset(discovery_reply,0x00,sizeof(discovery_reply));
                    discovery_reply[0]=(base_sequence>>24)&0xFF;
                    discovery_reply[1]=(base_sequence>>16)&0xFF;
                    discovery_reply[2]=(base_sequence>>8)&0xFF;
                    discovery_reply[3]=base_sequence&0xFF;
                    discovery_reply[4]=state+2;  // 2=idle 3=active
                    discovery_reply[5]=mac_address[0];
                    discovery_reply[6]=mac_address[1];
                    discovery_reply[7]=mac_address[2];
                    discovery_reply[8]=mac_address[3];
                    discovery_reply[9]=mac_address[4];
                    discovery_reply[10]=mac_address[5];
                    discovery_reply[11]=6; // hermes lite
                    discovery_reply[12]=100; // version 1.00
                    discovery_reply[20]=8; // receivers
                    discovery_reply[21]=1; // phase word

                    if ((rc=sendto(new_socket, discovery_reply, 60, 0,
                               (struct sockaddr*)&read_addr, sizeof(read_addr))) < 0) {
                        fprintf(stderr, "new_read_thread: Error sendtoa discovery_reply: %d",rc);
                        exit(EXIT_FAILURE);
                    }
                    break;
                case 3:  // set IP address
                    // ignore
                    break;
                case 4:  // erase
                    // ignore
                    break;
                case 5:  // program
                    // ignore
                    break;
                default:
                    fprintf(stderr, "new_read_thread: unknown command on port 1024: %d\n", command);
                    break;
            }
        
        }
            

    }

}

