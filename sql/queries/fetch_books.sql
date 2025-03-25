-- Mostrar título y blurb de todos los libros, junto con información secundaria
-- 
-- Para obtener el total de libros que cumplen con los criterios de esta consulta, puedes usar:
-- SELECT 
--     COUNT(DISTINCT b.book_id) AS total_books
-- FROM 
--     books b
-- LEFT JOIN reviews r ON b.book_id = r.book_id
-- WHERE
--     b.blurb IS NOT NULL AND b.blurb != '';

-- Mostrar título y blurb de todos los libros, junto con información secundaria
SELECT 
    b.book_id,
    b.title AS book_title,
    GROUP_CONCAT(DISTINCT a.name ORDER BY a.name ASC SEPARATOR ', ') AS authors, -- Autores (secundario)
    COUNT(DISTINCT t.tag_id) AS tag_count,  
    GROUP_CONCAT(DISTINCT t.name ORDER BY t.name ASC SEPARATOR ', ') AS tags,    -- Etiquetas (secundario)
    b.blurb
FROM 
    books b
LEFT JOIN book_authors ba ON b.book_id = ba.book_id
LEFT JOIN authors a ON ba.author_id = a.author_id
LEFT JOIN book_tags bt ON b.book_id = bt.book_id
LEFT JOIN tags t ON bt.tag_id = t.tag_id
WHERE
    b.blurb IS NOT NULL AND b.blurb != ''
GROUP BY
    b.book_id, b.title, b.blurb
ORDER BY
    -- b.title ASC;
    tag_count DESC;