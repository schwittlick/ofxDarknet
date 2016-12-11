#ifndef LIST_H
#define LIST_H

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list1{
    int size;
    node *front;
    node *back;
} list1;

list1 *make_list();
int list_find( list1 *l, void *val);

void list_insert( list1 *, void *);

void **list_to_array( list1 *l);

void free_list( list1 *l);
void free_list_contents( list1 *l);

#endif
