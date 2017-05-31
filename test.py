def get_tasks_names():
    tasks_names_in_labels_file = [
        'skin',
        'gender',
        'hair_cover',
        'hair_color',
        'hair_len',
        'hair_type',
        'hair_fringe',
        'beard',
        'glasses',
        #'face',
        #'mouth',
        #'nose',
        #'face_exp',
        #'brows',
        #'nose_type',
        #'nose_tip',
        'nose_width'
    ]
    tasks_names_to_work = [
        #'skin',
        #'gender',
        #'hair_cover',
        #'hair_color',
        #'hair_len',
        #'hair_type',
        #'hair_fringe',
        #'beard',
        #'glasses',
        #'face',
        #'mouth',
        #'nose',
        #'face_exp',
        #'brows',
        #'nose_type',
        #'nose_tip',
        'nose_width'
    ]
    return (tasks_names_in_labels_file, tasks_names_to_work)


microclasses_names = get_tasks_names()[0]
microclasses_names.extend(['count','filenames_list'])

microclasses_types = {'count' : int}
[microclasses_types.update({x: str}) for x in microclasses_names if x != 'count']

print microclasses_names
print microclasses_types