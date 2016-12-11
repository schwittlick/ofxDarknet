#pragma once 

#include <stdlib.h>
#include <string.h>
#include "list.h"

list1 *make_list()
{
	list1 *l = ( list1*)malloc(sizeof(list1));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list1 *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

void list_insert(list1 *l, void *val)
{
	node *new1 = (node*)malloc(sizeof(node));
	new1->val = val;
	new1->next = 0;

	if(!l->back){
		l->front = new1;
		new1->prev = 0;
	}else{
		l->back->next = new1;
		new1->prev = l->back;
	}
	l->back = new1;
	++l->size;
}

void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list1 *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(list1 *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

void **list_to_array(list1 *l)
{
    void **a = (void**)calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
