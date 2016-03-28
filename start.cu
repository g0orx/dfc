#include <stdio.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/ether.h>
#include <linux/if_packet.h>
#include <unistd.h>

#include "common.cuh"
#include "start.cuh"


static int sockfd;
static struct ifreq if_idx;
static struct ifreq if_mac;

void sendStart(int adc) {
    char buffer[75];
    struct ether_header *eh = (struct ether_header *) buffer;
    int header_length=sizeof(struct ether_header);
    struct sockaddr_ll socket_address;

    fprintf(stderr,"initStart: adc=%d\n",adc);

    // open raw socket
    if ((sockfd = socket(AF_PACKET, SOCK_RAW, IPPROTO_RAW)) == -1) {
        perror("initStart: socket");
        exit(-1);
    }

    // get the interface to send on
    memset(&if_idx, 0, sizeof(struct ifreq));
    strncpy(if_idx.ifr_name, interface, IFNAMSIZ-1);
    if (ioctl(sockfd, SIOCGIFINDEX, &if_idx) < 0) {
        perror("initStart: SIOCGIFINDEX");
        exit(-1);
    }

    // get the mac address
    memset(&if_mac, 0, sizeof(struct ifreq));
    strncpy(if_mac.ifr_name, interface, IFNAMSIZ-1);
    if (ioctl(sockfd, SIOCGIFHWADDR, &if_mac) < 0) {
        perror("initStart: SIOCGIFHWADDR");
        exit(-1);
    }

    // construct the buffer packet
    memset(buffer,0,sizeof(buffer));

    // destination address
    eh->ether_dhost[0]=0xFF;
    eh->ether_dhost[1]=0xFF;
    eh->ether_dhost[2]=0xFF;
    eh->ether_dhost[3]=0xFF;
    eh->ether_dhost[4]=0xFF;
    eh->ether_dhost[5]=0xFF;
  
    // source address
    eh->ether_shost[0]=((char *)&if_mac.ifr_hwaddr.sa_data)[0];
    eh->ether_shost[1]=((char *)&if_mac.ifr_hwaddr.sa_data)[1];
    eh->ether_shost[2]=((char *)&if_mac.ifr_hwaddr.sa_data)[2];
    eh->ether_shost[3]=((char *)&if_mac.ifr_hwaddr.sa_data)[3];
    eh->ether_shost[4]=((char *)&if_mac.ifr_hwaddr.sa_data)[4];
    eh->ether_shost[5]=((char *)&if_mac.ifr_hwaddr.sa_data)[5];

    // ether type
    eh->ether_type = htons(0xE000);

    socket_address.sll_ifindex = if_idx.ifr_ifindex;
    socket_address.sll_halen = ETH_ALEN;
    socket_address.sll_addr[0] = 0xFF;
    socket_address.sll_addr[1] = 0xFF;
    socket_address.sll_addr[2] = 0xFF;
    socket_address.sll_addr[3] = 0xFF;
    socket_address.sll_addr[4] = 0xFF;
    socket_address.sll_addr[5] = 0xFF;

    buffer[header_length]=(char)(adc&0xFF); // 0=ADC0, 5=ADC1, 3=21.073MHz sine wave
 
    fprintf(stderr,"initStart: send start: adc=%d\n",adc);
    if (sendto(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr*)&socket_address, sizeof(struct sockaddr_ll)) < 0) {
        printf("initStart: Send failed\n");
    }

    
    close(sockfd);
}
